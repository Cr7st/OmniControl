import pickle as pkl
import numpy as np
import torch

from utils import rotation_conversions
from utils import bvh, skeleton
from .dataset import Dataset
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot

lafan1_action_dict_9 = {
    0: 'aiming',
    1: 'dance',
    2: 'fallAndGetUp',
    3: 'fight',
    4: 'ground',
    5: 'jumps',
    6: 'obstacles',
    7: 'run',
    8: 'walk',
}

lafan1_parent_tree = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

lafan1_skeleton_offset = np.array([
    [0.0, 0.0, 0.0],
    [0.103458, 1.857834, 10.548502],
    [43.500008, -0.000028, -0.000002],
    [42.372200, -0.000011, -0.000010],
    [17.300005, 0.000001, 0.000004],
    [0.103456, 1.857835, -10.548497],
    [43.500038, -0.000031, 0.000015],
    [42.372269, -0.000019, 0.000010],
    [17.299995, -0.000004, 0.000006],
    [6.901966, -2.603724, 0.000006],
    [12.588104, -0.000002, 0.000002],
    [12.343198, 0.000013, -0.000014],
    [25.832897, -0.000017, -0.000002],
    [11.766611, 0.000019, 0.000005],
    [19.745913, -1.480379, 6.000116],
    [11.284129, 0.000004, -0.000028],
    [33.000038, 0.000006, 0.000033],
    [25.200012, -0.000007, -0.000007],
    [19.746098, -1.480396, -6.000068],
    [11.284148, -0.000029, -0.000023],
    [33.000088, 0.000013, 0.000010],
    [25.199785, 0.000162, 0.000438],
])

class Lafan1(Dataset):
    dataname = "lafan1"

    def __init__(self, datapath="/home/zheng/Code/KeyframeExtractor/datasets/lafan1", split='train',
                 control_joint=0, density=100, **kwargs):
        self.datapath = datapath

        super().__init__(control_joint=control_joint, density=density, **kwargs)

        data_paths = list(Path(self.datapath).rglob('*.bvh'))
        self._pose = []
        self._num_frames_in_video = []
        self._joints = []
        self._actions = []
        if split == 'test':
            data_paths = [path for path in data_paths if 'subject5' in path.__str__()]
        elif split == 'train':
            data_paths = [path for path in data_paths if 'subject5' not in path.__str__()]
        elif split == 'all':
            data_paths = data_paths

        # load from bvh first
        total_frame = 0
        sk = skeleton.SkeletonMotion()
        for path in data_paths:
            label = path.stem.split('_')[0][:-1]
            if 'push' in label or 'multipleActions' in label:
                continue
            elif 'fight' in label:
                label = 'fight'
            elif 'sprint' in label:
                label = 'run'
            sk.from_bvh(path)
            total_frame += sk.joints_global_positions.shape[0]
            for i in range(0, sk.joints_global_positions.shape[0] - self.num_frames + 1, self.num_frames):
                self._actions.append(label)
                self._joints.append(sk.joints_global_positions[i:i + self.num_frames] / 100) # cm to m
                self._num_frames_in_video.append(self.num_frames)
                rotations_ruler = np.stack([j.rotation[i:i + self.num_frames] for j in sk.joints], axis=1)
                self._pose.append(sRot.from_euler('xyz', rotations_ruler.reshape(-1, 3), degrees=True
                                                    ).as_rotvec().reshape(-1, 22*3))
        print(f'total frame num of bvh: {total_frame}')
        # then load from npz
        data_paths = list(Path(self.datapath).rglob('*.npz'))
        if split == 'test':
            data_paths = [path for path in data_paths if 'subject5' in path.__str__()]
        elif split == 'train':
            data_paths = [path for path in data_paths if 'subject5' not in path.__str__()]
        elif split == 'all':
            data_paths = data_paths
        for path in data_paths:
            label = path.stem.split('_')[0][:-1]
            if 'push' in label or 'multipleActions' in label:
                continue
            elif 'fight' in label:
                label = 'fight'
            elif 'sprint' in label:
                label = 'run'
            self._actions.append(label)
            data = np.load(path)
            self._joints.append(data['g_positions'] / 100) # cm to m
            self._num_frames_in_video.append(data['g_positions'].shape[0])
            rotations_6d = torch.from_numpy(data['rotations'])
            rotations_mat = rotation_conversions.rotation_6d_to_matrix(rotations_6d)
            rotations = rotation_conversions.matrix_to_axis_angle(rotations_mat)
            self._pose.append(rotations.reshape(-1, 22*3))


        total_num_actions = 9
        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = lafan1_action_dict_9

        reverse_dict = {v: k for k, v in lafan1_action_dict_9.items()}
        for i in range(len(self._actions)):
            self._actions[i] = reverse_dict[self._actions[i]]

    def _load_joints3D(self, ind, frame_idx):
        return self._joints[ind][frame_idx]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose
