# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.a2m.lafan1 import lafan1_parent_tree, lafan1_skeleton_offset
import shutil
from data_loaders.tensors import collate
from utils.text_control_example import collate_all
from os.path import join as pjoin
from utils.bvh import Bvh
from scipy.spatial.transform import Rotation as sRot
import utils.rotation_conversions as geometry
from utils import heuristic


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['kit', 'humanml']:
        max_frames = 196
    elif args.dataset == 'lafan1':
        max_frames = 219
    else:
        max_frames = 60

    if args.dataset == 'kit':
        fps = 12.5
    elif args.dataset == 'lafan1':
        fps = 30
    else:
        fps = 20
    n_frames = max_frames
    is_using_data = not any([args.text_prompt, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')

    hints = None
    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        if args.text_prompt == 'predefined':
            # generate hint and text
            texts, hints = collate_all(n_frames, args.dataset)
            args.num_samples = len(texts)
            if args.cond_mode == 'only_spatial':
                # only with spatial control signal, and the spatial control signal is defined in utils/text_control_example.py
                texts = ['' for i in texts]
            elif args.cond_mode == 'only_text':
                # only with text prompt, and the text prompt is defined in utils/text_control_example.py
                hints = None
        else:
            # otherwise we use text_prompt
            texts = [args.text_prompt]
            args.num_samples = 1
            hint = None
    elif args.action_name != '':
        action_text = [args.action_name]
        args.num_samples = 1
        hints = None

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = get_dataset_loader(name=args.dataset,
                              datapath=args.inpaint_motions,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='train',
                              hml_mode='test')
    data.shuffle = False
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.text_prompt])
        if is_t2m:
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        if hints is not None:
            collate_args = [dict(arg, hint=hint) for arg, hint in zip(collate_args, hints)]

        _, model_kwargs = collate(collate_args)

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_rotations = []
    all_keyframes = []
    all_rotations_6d = []
    all_hint = []
    all_hint_for_vis = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames if not is_using_data else max_frames),
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
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample, rotations, global_orient = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
                                                         glob=True, translation=True,
                                                         jointstype='smpl', vertstrans=True, betas=None, beta=0,
                                                         glob_rot=None,
                                                         get_rotations_back=True)
        nsamples, time, n_joints, feats = rotations.shape
        all_rotations_6d.append(rotations.cpu().numpy())
        rotations = geometry.rotation_6d_to_matrix(rotations).cpu().numpy()
        rotations = sRot.from_matrix(rotations.reshape(-1, 3, 3)).as_euler('xyz', degrees=True).reshape(nsamples, time,
                                                                                                        n_joints, 3)

        for i in range(nsamples):
            keyframes, _ = heuristic.keyframe_jerk(sample[i].permute(2, 0, 1), 30, 30, smooth_window=3, nms=True,
                                                   nms_threshold=0.85)
            keyframes = keyframes.tolist()
            keyframes = [i for i in keyframes if time - 1 > i > 9]
            keyframes += [9, time - 1]
            keyframes.sort()
            all_keyframes.append(keyframes)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

            if 'hint' in model_kwargs['y']:
                hint = model_kwargs['y']['hint']
                # denormalize hint
                if args.dataset == 'humanml':
                    spatial_norm_path = './dataset/humanml_spatial_norm'
                elif args.dataset == 'kit':
                    spatial_norm_path = './dataset/kit_spatial_norm'
                elif args.dataset == 'lafan1':
                    spatial_norm_path = './dataset/lafan1_spatial_norm'
                else:
                    raise NotImplementedError('unknown dataset')
                raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))).cuda()
                raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))).cuda()
                mask = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(-1) != 0
                hint = hint * raw_std + raw_mean
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask.unsqueeze(-1)
                hint = hint.view(hint.shape[0], hint.shape[1], -1)
                # ---
                all_hint.append(hint.data.cpu().numpy())
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3)
                all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        all_rotations.append(rotations)

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_rotations = np.concatenate(all_rotations, axis=0)
    all_rotations = all_rotations[:total_num_samples]
    all_rotations_6d = np.concatenate(all_rotations_6d, axis=0)
    all_rotations_6d = all_rotations_6d[:total_num_samples]
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if 'hint' in model_kwargs['y']:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    
    if len(all_hint) != 0:
        from utils.simple_eval import simple_eval
        results = simple_eval(all_motions, all_hint, n_joints)
        print(results)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths, "hint": all_hint_for_vis,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    if args.dataset == 'kit':
        skeleton = paramUtil.kit_kinematic_chain
    elif args.dataset == 'lafan1':
        skeleton = paramUtil.lafan1_kinematic_chain
    else:
        skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            keyframes = all_keyframes[rep_i * args.batch_size + sample_i]
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            rotations = all_rotations[rep_i * args.batch_size + sample_i][:length]
            rotations_6d = all_rotations_6d[rep_i * args.batch_size + sample_i][:length]
            if 'hint' in model_kwargs['y']:
                hint = all_hint_for_vis[rep_i*args.batch_size + sample_i]
            else:
                hint = None
            save_file = f'{caption}_rep{rep_i}'
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            # plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

            if args.dataset == 'lafan1':
                bvh = Bvh()
                l_positions = np.stack([lafan1_skeleton_offset] * length, axis=0)
                l_positions[:, 0] = motion[:, 0] * 100
                bvh.load_from_data(l_positions, rotations, lafan1_parent_tree, fps=30)
                bvh.save(animation_save_path + '.bvh')
                np.savez(animation_save_path + '.npz',
                         l_positions=l_positions, rotations=rotations_6d, keyframes=np.array(keyframes))

        # sample_files = save_multiple_samples(args, out_path,
        #                                        row_print_template, all_print_template, row_file_template, all_file_template,
        #                                        caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
