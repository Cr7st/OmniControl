import gc
import os
import json

import numpy as np
import torch

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader, get_dataset
from utils.model_util import create_model_and_diffusion

# dataloader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=210)
dataset = get_dataset('lafan1', 210, 'train', 'train',
                      0, 100, "")

raw_data_list = []
data_list = []

from tqdm import tqdm
for out in tqdm(dataset):
    data = out['inp'].permute(2, 0, 1)
    data = data.reshape(data.shape[0], -1).cpu()

    data_list.append(data)
    raw_data_list.append(torch.from_numpy(out['hint'].reshape(-1, 66)).cpu())

raw_data = torch.cat(raw_data_list).numpy()
data = torch.cat(data_list).numpy()

# del raw_data_list
# del data_list

gc.collect()

mean_raw = raw_data.mean(axis=0)
print('mean raw')
std_raw = raw_data.std(axis=0)
print('std raw')

mean = data.mean(axis=0)
print('mean')
std = data.std(axis=0)
print('std')

np.save('./dataset/lafan1_spatial_norm/Mean.npy', mean)
np.save('./dataset/lafan1_spatial_norm/Std.npy', std)
np.save('./dataset/lafan1_spatial_norm/Mean_raw.npy', mean_raw)
np.save('./dataset/lafan1_spatial_norm/Std_raw.npy', mean_raw)
