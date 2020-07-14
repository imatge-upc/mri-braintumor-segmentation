import importlib

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from src.dataset.loaders.brats_dataset import BratsDataset
from src.dataset.utils import dataset
from src.dataset.utils.visualization import plot_3_view, plot_batch_slice
from matplotlib import pyplot as plt
import numpy as np


def matplotlib_imshow(images, normalized=False):

    img = torchvision.utils.make_grid(images, padding=10)
    npimg = img.numpy()
    # c h w
    trans_im = np.transpose(npimg, (1, 2, 0))
    plt.imshow(trans_im, cmap="gray")
    plt.savefig("batch")

csv = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/brats20_data.csv"
data, data_test = dataset.read_brats(csv)

modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                     BratsDataset.t1ce_idx: True}
sampling_method = importlib.import_module("src.dataset.patching.random_tumor_distribution")
transforms = T.Compose([T.ToTensor()])

data_train = data * 100
batch_size = 4
train_dataset = BratsDataset(data_train, modalities_to_use, sampling_method, (128, 128, 128), transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


i = 0
s = 20
seg_maps_2d, volumes_2d = [], []
for patients_ids, data_batch, labels_batch in train_loader:
    for seg_map, volume in zip(labels_batch, data_batch):
        slice = seg_map[:, s, :].unsqueeze(0)
        seg_maps_2d.append(slice)

       #  volume_slice = volume_mod[0, :, s, :].unsqueeze(0)
        volume_slice = torch.cat((volume[0, :, s, :].unsqueeze(0),
                                  volume[1, :, s, :].unsqueeze(0),
                                  volume[2, :, s, :].unsqueeze(0)), 0)

        volumes_2d.append(volume_slice)


    seg_maps_2d_tensor = torch.stack(seg_maps_2d)
    volumes_2d_tensor = torch.stack(volumes_2d)
    matplotlib_imshow(seg_maps_2d)
    print(f"Batch {i}")
    i +=1
    break


