from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
import torch
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy('file_system')

from eval.a2m.FID import FIDModule
from eval.a2m.penetration import PenetrationEvaluator, FootSkateEvaluator, TrajErrorEvaluator
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import data_utils
from scipy.spatial.transform import Rotation as sRot
import utils.rotation_conversions as geometry
import lightning as L
import subprocess
import os
from tqdm import tqdm
from utils import heuristic

run_mib = False

def run_cmd(cmd, verbo=True, bg=False):
    if verbo: print('[run] ' + cmd, 'run')
    if bg:
        args = cmd.split()
        # print(args)
        p = subprocess.Popen(args)
        return [p]
    else:
        exit_status = os.system(cmd)
        if exit_status != 0:
            raise RuntimeError
        return []

class EvaluateWrapper(Dataset):
    def __init__(self, l_position = None):
        super().__init__()
        self.l_position = l_position
        self.max_frames = 0
        self.states = []
        self.rotations = []
        self.g_positions = []
        self.keyframes = []
        self.actions = []
        self.metric_name = None
        self.traj = []

    def append_data(self, rotations, root_trans, g_positions, keyframes, action, traj):
        intervals = [keyframes[i] - keyframes[i - 1] for i in range(1, len(keyframes))]
        intervals.append(0)
        intervals.sort()
        intervals = np.array(intervals).astype(np.float32)[:, None]
        self.max_frames = len(keyframes) if len(keyframes) > self.max_frames else self.max_frames
        self.rotations.append(rotations)
        self.g_positions.append(g_positions)
        self.keyframes.append(np.array(keyframes))
        self.actions.append(action)
        self.traj.append(traj)

    def on_append_data_end(self):
        if not run_mib:
            return

        full_rotations_list = np.zeros((len(self), 219, 22, 6), dtype=np.float32)
        full_positions_list = np.zeros((len(self), 219, 22, 3), dtype=np.float32)
        assert np.isnan(full_positions_list).sum() == 0
        assert np.isnan(self.l_position).sum() == 0
        full_positions_list[:, :] = self.l_position
        assert np.isnan(full_positions_list).sum() == 0

        keyframes_list = [ np.concatenate([ks, np.zeros((self.max_frames - ks.shape[0]), dtype=np.int32)], axis=-1, dtype=np.int32)[None] for ks in self.keyframes]
        keyframes_list = np.concatenate(keyframes_list, axis=0)
        action_list = np.concatenate(self.actions, axis=0).reshape(-1, 1)

        for i, ks in enumerate(self.keyframes):
            full_rotations_list[i, ks] = self.rotations[i]
            full_positions_list[i, ks, 0] = self.g_positions[i][:, 0]
            assert np.isnan(self.g_positions[i]).sum() == 0
            full_rotations_list[i, :ks[0]] = self.rotations[i][0]
            full_positions_list[i, :ks[0], 0] = self.g_positions[i][0, 0]

        mib_eval_path = os.path.join('./tmp_res', f"eval_keyframe.npz")
        np.savez(mib_eval_path, rotations=full_rotations_list, l_positions=full_positions_list, keyframes=keyframes_list, action=action_list)

        cmd = f"~/environment/bin/python3 {os.path.join('~/Code/staging/RSMT', 'eval.py')}"
        cmd += f" --data_path {mib_eval_path}"
        cmd += f" --output_path './tmp_res'"
        cmd += f" --output_name eval_motion.npz"
        cmd += f" --gpus 0"
        run_cmd(cmd)

        eval_path = os.path.join('./tmp_res', 'eval_motion.npz')
        self._load_mib_data(eval_path)

    def _load_mib_data(self, path, mirror = False):
        """

        Args:
            path (Path): path to the npz file
            g_position (np.ndarray): (batch_size, n_frames, 22, 3)
            rotation (np.ndarray): (batch_size, n_frames, 22, 6)
        """
        data = np.load(path, allow_pickle=True)
        g_positions = data['g_positions']
        rotations = data['rotations']
        action = data['action']
        # keyframes = data['keyframes']
        batch_size = g_positions.shape[0]
        length = g_positions.shape[1]
        self.max_frames = length if length > self.max_frames else self.max_frames

        st = np.arange(0, batch_size) * length
        ed = np.arange(1, batch_size+1) * length
        seq_idx = np.stack((st, ed), axis=-1)

        self.keyframes = seq_idx
        self.g_positions = g_positions.reshape(-1, 22, 3)
        self.rotations = rotations.reshape(-1, 22, 6)
        # self._clip_label = [self.action_to_label[action[i][0]] for i in range(batch_size)]
        return 0

    def clear(self):
        self.max_frames = 0
        self.states.clear()
        self.rotations.clear()
        self.g_positions.clear()
        self.keyframes.clear()
        self.actions.clear()

    def __getitem__(self, idx):
        if run_mib:
            start_idx, end_idx = self.keyframes[idx]
            interval = end_idx - start_idx
            g_positions = self.g_positions[start_idx : end_idx].copy()
            rotations = self.rotations[start_idx : end_idx].copy()
        else:
            keyframes = self.keyframes[idx]
            interval = len(keyframes)
            g_positions = self.g_positions[idx].copy()
            rotations = self.rotations[idx].copy()

        # pos 移到中心, 并转为正面
        trans_offset = g_positions[0, 0, ::2]
        g_positions[..., ::2] -= trans_offset

        g_positions, rotations = data_utils.rotate_start_to_v2(g_positions, rotations, frame=0)

        # if self.data_repr == 'gpos':
        states = np.concatenate([g_positions.reshape(interval, -1)])
        # elif self.data_repr == 'rot6d_pos':
        # states = np.concatenate([rotations.reshape(len(keyframes), -1), g_positions[:, 0]], axis=1)

        if self.metric_name == 'FID' or self.metric_name == 'MiB':
            zero_padding = np.zeros((self.max_frames - interval, states.shape[1]), dtype=np.float32)
            states = np.concatenate([states, zero_padding], axis=0)
            return states.astype(np.float32), self.actions[idx].astype(np.float32), interval
        elif self.metric_name == 'penetration' or self.metric_name == 'foot skate':
            if interval < self.max_frames:
                padding = [np.zeros_like(g_positions[0][None]) for _ in range(self.max_frames - interval)]
                g_positions = np.concatenate([g_positions] + padding, axis=0)
            return g_positions.astype(np.float32), interval
        elif self.metric_name == 'traj':
            return self.traj[idx].astype(np.float32), g_positions[:, 0].astype(np.float32)


    def __len__(self):
        return len(self.keyframes)

    def calc_stats(self):
        for i in range(len(self)):
            states, _, length = self[i]
            if i == 0:
                all_states = states[:length]
            else:
                all_states = np.concatenate([all_states, states[:length]], axis=0)
        self.mean = all_states.mean(axis=0)[None]
        self.std = all_states.std(axis=0)[None]
        # self.std[np.where(self.std < 1e-3)] = 1e-3
        return self.mean, self.std


def sample2eval(sample, model_kwargs, full=False):
    n_frames = sample.shape[-1]
    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size,
                                                                                            n_frames).bool()
    sample, rotations, global_orient = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
                                                     glob=True, translation=True,
                                                     jointstype='smpl', vertstrans=True, betas=None, beta=0,
                                                     glob_rot=None,
                                                     get_rotations_back=True)
    nsamples, time, njoints, feats = rotations.shape
    root_trans = sample[:, 0, ...].permute(0, 2, 1) * 100
    rotations_list = []
    root_trans_list = []
    g_positions_list = []
    keyframes_list = []
    actions = []
    for i in range(nsamples):
        if full is False:
            keyframes, _ = heuristic.keyframe_jerk(sample[i].permute(2, 0, 1), 30, 30, smooth_window=3,
                                                   random_infill=True, nms_threshold=0.85)
            keyframes = keyframes.tolist()
            keyframes = [i for i in keyframes if time - 1 > i > 9]
            keyframes += [9, time - 1]
            keyframes.sort()
        else:
            keyframes = list(range(9, time-1))
        rotations_list.append(rotations[i, keyframes])
        root_trans_list.append(root_trans[i, keyframes])
        g_positions_list.append(sample[i, :, :, keyframes].permute(2, 0, 1) * 100)
        keyframes_list.append(keyframes)
        actions.append(model_kwargs['y']['action'][i].squeeze())
    return rotations_list, root_trans_list, g_positions_list, keyframes_list, actions


if __name__ == '__main__':
    max_frames = 210
    fid_evaluator = FIDModule.load_from_checkpoint('/home/zheng/Code/KeyframeGenerator/exps/MotionFID/full_motion_v2/checkpoint/epoch=54-train_acc=0.759-val_acc=0.760.ckpt')
    penetration_evaluator = PenetrationEvaluator()
    penetration_evaluator = PenetrationEvaluator()
    foot_skate_evaluator = FootSkateEvaluator()
    traj_err_evaluator = TrajErrorEvaluator()
    fid_evaluator.mean = np.load('/home/zheng/Code/KeyframeGenerator/data/model_stats/FID/mean.npy')
    fid_evaluator.std = np.load('/home/zheng/Code/KeyframeGenerator/data/model_stats/FID/std.npy')

    eval_wrapper = EvaluateWrapper()

    args = evaluation_parser()
    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=210, split=split,
                                   hml_mode='gt', datapath='/home/zheng/Code/KeyframeGenerator/datasets/lafan1_keyframes')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=210, split=split,
                                    hml_mode='eval', datapath='/home/zheng/Code/KeyframeGenerator/datasets/lafan1_keyframes')
    num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()
    # for data in tqdm(gen_loader):
    #     input_motions, model_kwargs = data
    #     input_motions = input_motions.to(dist_util.dev())
    #     rotations, root_trans_list, g_positions, keyframes_list, actions = sample2eval(input_motions, model_kwargs)
    #     for i in range(len(rotations)):
    #         eval_wrapper.append_data(rotations[i].cpu().numpy(), root_trans_list[i].cpu().numpy(),
    #                                  g_positions[i].cpu().numpy(), keyframes_list[i],
    #                                  actions[i].cpu().numpy())
    #
    # dataloader = DataLoader(eval_wrapper, batch_size=64, shuffle=False)
    trainer = L.Trainer()
    # trainer.predict(fid_evaluator, dataloader)
    fid_evaluator.gt_mu = np.load('/home/zheng/Code/KeyframeGenerator/fid_mu.npy')
    fid_evaluator.gt_sigma = np.load('/home/zheng/Code/KeyframeGenerator/fid_sigma.npy')
    for i in range(5):
        eval_wrapper = EvaluateWrapper()

        for data in gen_loader:
            _, model_kwargs = data
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            for k, v in model_kwargs['y'].items():
                if torch.is_tensor(v):
                    model_kwargs['y'][k] = v.to(dist_util.dev())

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample[:, :263]

            rotations, root_trans_list, g_positions, keyframes_list, actions = sample2eval(sample, model_kwargs, full=True)
            for i in range(len(rotations)):
                eval_wrapper.append_data(rotations[i].cpu().numpy(), root_trans_list[i].cpu().numpy(),
                                         g_positions[i].cpu().numpy(), keyframes_list[i], actions[i].cpu().numpy(),
                                         model_kwargs['hint'][:, -1].permute(2, 0, 1))

        dataloader = DataLoader(eval_wrapper, batch_size=64, shuffle=False)
        result = fid_evaluator.evaluate(dataloader)
        print(f'fid: {result["FID"]}, acc： {result["acc"]}')
        result = penetration_evaluator.evaluate(dataloader)
        print(f'penetration: {result["penetration"]}')
        result = foot_skate_evaluator.evaluate(dataloader)
        print(f'foot skate: {result["skating ratio"]}')
        result = traj_err_evaluator.evaluate(dataloader)
        print(f'traj error 0.2: {result["traj fail 0.2"]}')
        result = traj_err_evaluator.evaluate(dataloader)
        print(f'traj error 0.5: {result["traj fail 0.5"]}')


