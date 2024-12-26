from .FID import Evaluator
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch

class PenetrationEvaluator(Evaluator):
    def __init__(self, threshold=-3):
        super().__init__()
        self.threshold = threshold
        self.name = 'penetration'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        total_entry = 0
        total_metric = 0
        for idx, data in enumerate(dataloader):
            g_positions, lengths = data[0], data[1]
            y_value = g_positions[..., 1]
            mask = y_value < self.threshold
            y_value[~mask] = 0

            total_entry += g_positions.shape[0]
            total_metric += (y_value.sum(dim=(1, 2)) / lengths).sum()

        result = -total_metric / total_entry
        return {
            'penetration': result
        }

class FootSkateEvaluator(Evaluator):
    def __init__(self, threshold_height=5, threshold_vel=20):
        super().__init__()
        self.threshold_height = threshold_height
        self.threshold_vel = threshold_vel
        self.name = 'foot skate'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        skating_ratio_sum = 0
        all_size = 0
        for idx, data in enumerate(dataloader):
            fps = 30.0
            avg_window = 5  # frames
            g_positions = data[0]

            batch_size = g_positions.shape[0]
            all_size += batch_size
            # 8 left, 4 right foot. XZ plane, y up
            # motions [bs, 22, 3, max_len]
            motions = g_positions.permute(0, 2, 3, 1)
            verts_feet = motions[:, [8, 4], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
            verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
                                                  axis=2) * fps  # [bs, 2, max_len-1]
            # [bs, 2, max_len-1]
            vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

            verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
            # If feet touch ground in agjecent frames
            feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < self.threshold_height),
                                          (verts_feet_height[:, :, 1:] < self.threshold_height))  # [bs, 2, max_len - 1]
            # skate velocity
            skate_vel = feet_contact * vel_avg

            # it must both skating in the current frame
            skating = np.logical_and(feet_contact, (verts_feet_plane_vel > self.threshold_vel))
            # and also skate in the windows of frames
            skating = np.logical_and(skating, (vel_avg > self.threshold_vel))

            # Both feet slide
            skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
            skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
            skating_ratio_sum += skating_ratio

        return {
            'skating ratio': torch.tensor(skating_ratio_sum.sum() / all_size)
        }


class TrajErrorEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.name = 'traj'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        all_size = 0
        traj_fail_02_sum = 0
        traj_fail_05_sum = 0
        for idx, data in enumerate(dataloader):
            gt_traj, traj = data[0], data[1]
            all_size += traj.shape[0]
            gt_traj = gt_traj.detach().cpu().numpy()
            traj = traj.detach().cpu().numpy()
            dist_error = np.linalg.norm(gt_traj - traj, axis=-1)
            traj_fail_02 = 1.0 - (dist_error <= 20).all()
            traj_fail_05 = 1.0 - (dist_error <= 50).all()
            traj_fail_02_sum += traj_fail_02
            traj_fail_05_sum += traj_fail_05
        return {
            'traj fail 0.2': torch.tensor(traj_fail_02_sum / all_size),
            'traj fail 0.5': torch.tensor(traj_fail_05_sum / all_size)
        }



