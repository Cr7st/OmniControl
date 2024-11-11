import random

import numpy as np
import torch
# from utils.action_label_to_idx import action_label_to_idx
from data_loaders.tensors import collate
from os.path import join as pjoin
from utils.misc import to_torch
import utils.rotation_conversions as geometry


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train",
                 pose_rep="rot6d", translation=True, glob=True, max_len=-1, min_len=-1, num_seq_max=-1, **kwargs):
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max

        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)
        self.use_action_cat_as_text_labels = kwargs.get('use_action_cat_as_text_labels', False)
        self.only_60_classes = kwargs.get('only_60_classes', False)
        self.leave_out_15_classes = kwargs.get('leave_out_15_classes', False)
        self.use_only_15_classes = kwargs.get('use_only_15_classes', False)
        self.mode = kwargs.get('mode', 'train')
        self.control_joint = kwargs.get('control_joint', 0)
        self.density = kwargs.get('density', 100)

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()

        # to remove shuffling
        self._original_train = None
        self._original_test = None

        if self.dataname == 'lafan1':
            spatial_norm_path = './dataset/lafan1_spatial_norm'
            data_root = './dataset/lafan1_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')

        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_pose_data(self, data_index, frame_ix):
        pose, hint = self._load(data_index, frame_ix)
        label = self.get_label(data_index)
        return pose, hint, label

    def get_label(self, ind):
        action = self.get_action(ind)
        return self.action_to_label(action)

    def get_action(self, ind):
        return self._actions[ind]

    def action_to_action_name(self, action):
        return self._action_classes[action]

    def action_name_to_action(self, action_name):
        # self._action_classes is either a list or a dictionary. If it's a dictionary, we 1st convert it to a list
        all_action_names = self._action_classes
        if isinstance(all_action_names, dict):
            all_action_names = list(all_action_names.values())
            assert list(self._action_classes.keys()) == list(range(len(all_action_names)))  # the keys should be ordered from 0 to num_actions

        sorter = np.argsort(all_action_names)
        actions = sorter[np.searchsorted(all_action_names, action_name, sorter=sorter)]
        return actions

    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]

        # inp, target = self._get_item_data_index(data_index)
        # return inp, target
        return self._get_item_data_index(data_index)

    def random_mask(self, joints, n_joints=22, density=1):
        if n_joints == 22:
            # humanml3d, lafan1
            controllable_joints = np.array([0]) if self.dataname == 'lafan1' else np.array([0, 10, 11, 15, 20, 21])
        else:
            # kit
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        choose_joint = [self.control_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        # density = 100
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0]) if self.dataname == 'lafan1' else np.array([0, 10, 11, 15, 20, 21])
        else:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        num_joints = len(controllable_joints)
        # joints: length, 22, 3
        num_joints_control = np.random.choice(num_joints, 1)
        # only use one joint during training
        num_joints_control = 1
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = controllable_joints[choose_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def _load(self, ind, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(ind, frame_ix)
                joints3D[..., ::2] = joints3D[..., ::2] - joints3D[0, 0, ::2]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                raise ValueError("Global joints positions must be available for OmniControl.")
                # if pose_rep == "xyz":
                #     raise ValueError("This representation is not possible.")
                # if getattr(self, "_load_translation") is None:
                #     raise ValueError("Can't extract translations.")
                # ret_tr = self._load_translation(ind, frame_ix)
                # ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                seq_len, n_joints = ret.shape[0], ret.shape[1]
                pose = self._load_rotvec(ind, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if self.align_pose_frontview:
                    first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(pose[0][0])
                    all_root_poses_matrix = geometry.axis_angle_to_matrix(pose[:, 0, :])
                    aligned_root_poses_matrix = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1),
                                                             all_root_poses_matrix)
                    pose[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix)

                    if self.translation:
                        ret = ret.reshape(-1, 3)
                        ret = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1).float(),
                                              torch.transpose(ret, 0, 1))
                        ret = torch.transpose(ret, 0, 1)
                        ret = ret.reshape(seq_len, n_joints, 3)
                        ret_tr = ret[:, 0, :]

                hint = self.random_mask_train(ret.numpy(), n_joints) if self.mode == 'train' else self.random_mask(ret, n_joints)
                hint = hint.reshape(hint.shape[0], -1)

                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float(), hint.float() if type(hint) == torch.Tensor else hint.astype(float)

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            if num_frames > nframes:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes),
                                               padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                if not shift == 0:
                    print('random selecting')
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        inp, hint, action = self.get_pose_data(data_index, frame_ix)


        output = {'inp': inp, 'action': action, 'hint': hint}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output


    def get_mean_length_label(self, label):
        if self.num_frames != -1:
            return self.num_frames

        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(self._actions[index] == action).squeeze(1)
        lengths = self._num_frames_in_video[np.array(index)[choices]]

        if self.max_len == -1:
            return np.mean(lengths)
        else:
            # make the lengths less than max_len
            lengths[lengths > self.max_len] = self.max_len
        return np.mean(lengths)

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
