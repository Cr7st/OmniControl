import numpy as np
import torch
import matplotlib.pyplot as plt


def smooth(y, box_pts=3):
    box = torch.ones(box_pts, device=y.device) / box_pts
    y_smooth = torch.nn.functional.conv1d(y.view(1, 1, -1), box.view(1, 1, -1), padding=box_pts // 2).view(-1)
    return y_smooth


def approximate_jerk(frames, fps):
    jkp = ((frames[3:] - 3 * frames[2:-1] + 3 * frames[1:-2] - frames[:-3]) * (fps ** 3)).norm(dim=-1)
    return jkp


def non_max_suppression_jerks(jerks, kf_rate, threshold=0.85):
    """
    jerks: torch.tensor, the jerk magnit ude of each frame
    kf_rate: int, the average number of frames between two keyframes
    threshold: float, the threshold for non-maximum suppression

    return: torch.tensor, the index of keyframes
    """
    # find supress the values that with closer index to max values
    if jerks.size(0) == 0:
        return torch.empty((0,), dtype=torch.long)
    # normalize the jerks
    jerks_normalized = jerks / jerks.max()
    # make sure that the first and last frames are selected as keyframes
    jerks_normalized[6] = jerks_normalized[-1] = 1.
    max_jerks_index = torch.argsort(jerks_normalized, descending=True)
    mean = jerks_normalized[max_jerks_index].mean()
    max_jerks_index = max_jerks_index[torch.where(jerks_normalized[max_jerks_index] > mean)[0]]

    ind = 0
    while ind < len(max_jerks_index):
        i = max_jerks_index[ind]
        close_index = max_jerks_index[torch.where(torch.abs(max_jerks_index - i) < (kf_rate // 2))[0]]
        corresponding_values = jerks_normalized[[close_index]]
        max_index = torch.argmax(corresponding_values)
        diff_jerk = torch.abs(corresponding_values - corresponding_values[max_index])

        # remove every one within the threshold, which is very close to the max value
        combine = (diff_jerk < threshold).bool() & (diff_jerk > 0).bool()
        combine_index = close_index[combine]
        rm_index = torch.isin(max_jerks_index, combine_index)

        if rm_index.shape[0] > 0:
            max_jerks_index = max_jerks_index[~rm_index]

        ind += 1

    return max_jerks_index


def keyframe_jerk(joint_positions, fps, keyframe_rate, smooth_window=3, nms=False, nms_threshold=0.85):
    if type(joint_positions) != torch.Tensor:
        joint_positions = torch.tensor(joint_positions)
    jerk_magnitude = approximate_jerk(joint_positions[:, 0, :], fps)
    jerk_magnitude = smooth(jerk_magnitude, smooth_window)

    if not nms:
        # select the highest jerk magnitude in each fps
        # seperate each block of keyframes
        blocks_length = len(joint_positions) // keyframe_rate
        keyframe_index_array = []
        for i in range(blocks_length):
            if i == blocks_length - 1:
                jerk_magnitude_block = jerk_magnitude[i * keyframe_rate:]
            else:
                jerk_magnitude_block = jerk_magnitude[i * keyframe_rate:(i + 1) * keyframe_rate]
            keyframe_index = torch.argmax(jerk_magnitude_block) + i * keyframe_rate
            keyframe_index_array.append(keyframe_index)
        keyframe_index_array = torch.tensor(keyframe_index_array)
    else:
        keyframe_index_array = non_max_suppression_jerks(jerk_magnitude, keyframe_rate, threshold=nms_threshold)

    # +3 make up for the 3 frames that were removed in the jerk calculation
    return keyframe_index_array + 3, jerk_magnitude


if __name__ == '__main__':
    dataset_root = r"./exps/guidance_on_action"
    dataset_root_case1 = r"{}/aiming1_subject1_730_948".format(dataset_root)

    for i, data_root in enumerate([dataset_root_case1]):
        # fig, ax = plt.subplots(2, 1, figsize=(24, 10))
        data = np.load(data_root + r".npz")

        fps = 30
        key_frames_gt = data['keyframes'] - 3
        framerate = len(data['g_positions']) // len(key_frames_gt)

        keyframe_jerk_nms, jerk_magnitude = keyframe_jerk(data['g_positions'], fps, framerate, smooth_window=3,
                                                          nms=True, nms_threshold=0.85)
        keyframe_jerk_block, jerk_magnitude = keyframe_jerk(data['g_positions'], fps, framerate, smooth_window=3,
                                                            nms=False, nms_threshold=0.85)

        joint_index = 0
        time_steps = np.arange(jerk_magnitude.shape[0])

        # ax[0].plot(time_steps, jerk_magnitude, label=f'Joint {joint_index} Jerk', c='black', linewidth=2)
        # ax[0].set_title(f'Jerk Magnitude over Time for Joint {joint_index}', fontsize=20)
        # ax[0].set_xticks(np.arange(0, jerk_magnitude.shape[0], 10), fontsize=15)
        # ax[0].set_ylabel('Jerk Magnitude', fontsize=15)
        #
        # # mark key frames
        # key_frames = data['keyframes']
        # # draw vertical lines
        # for key_frame in key_frames:
        #     ax[0].axvline(x=key_frame - 3, color='r', linestyle='--', linewidth=2)
        #     if key_frame == key_frames[0]:
        #         ax[0].axvline(x=key_frame - 3, color='r', linestyle='--', label='keyframe_gt', linewidth=2)
        # for key_frame in keyframe_jerk_nms:
        #     ax[0].axvline(x=key_frame, color='g', linestyle='--', linewidth=2)
        #     if key_frame == keyframe_jerk_nms[0]:
        #         ax[0].axvline(x=key_frame, color='g', linestyle='--', label='keyframe_nms', linewidth=2)
        # ax[0].legend(fontsize=20, loc='upper right')
        # ax[0].grid(True)
        # ax[0].set_title(f'Select by NMS', fontsize=20)
        #
        # ax[1].plot(time_steps, jerk_magnitude, label=f'Joint {joint_index} Jerk', c='black', linewidth=2)
        # ax[1].set_title(f'Jerk Magnitude over Time for Joint {joint_index}', fontsize=20)
        # ax[1].set_xticks(np.arange(0, jerk_magnitude.shape[0], 10), fontsize=15)
        # ax[1].set_ylabel('Jerk Magnitude', fontsize=15)
        #
        # for key_frame in key_frames:
        #     ax[1].axvline(x=key_frame - 3, color='r', linestyle='--', linewidth=2)
        #     if key_frame == key_frames[0]:
        #         ax[1].axvline(x=key_frame - 3, color='r', linestyle='--', label='keyframe_gt', linewidth=2)
        # for key_frame in keyframe_jerk_block:
        #     ax[1].axvline(x=key_frame, color='b', linestyle='--', linewidth=2)
        #     if key_frame == keyframe_jerk_block[0]:
        #         ax[1].axvline(x=key_frame, color='b', linestyle='--', label='keyframe_block', linewidth=2)
        # ax[1].legend(fontsize=20, loc='upper right')
        # ax[1].grid(True)
        # ax[1].set_title(f'Select by Block', fontsize=20)
        #
        # # set the title
        # fig.suptitle('Jerk Magnitude over Time for Joint 0 for {}'.format(data_root.split("\\")[-1]), fontsize=20)
        # fig.tight_layout()
        # fig.savefig("jerk_magnitude_{}.png".format(data_root.split("\\")[-1]))
        # plt.clf()

    print("Done")
