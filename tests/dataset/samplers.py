import importlib
import sys
from src.config import BratsConfiguration
from src.dataset.loaders.batch_sampler import BratsPatchSampler
from src.dataset.loaders.brats_dataset import BratsDataset
from src.dataset.utils import dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T


def test():


    csv = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/brats20_data.csv"
    data, data_test = dataset.read_brats(csv)

    modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                         BratsDataset.t1ce_idx: True}
    transforms = T.Compose([T.ToTensor()])
    sampling_method = importlib.import_module("src.dataset.patching.random_tumor_distribution")
    patch_size = (128, 128, 128)
    n_patches = 10

    data = data * n_patches
    train_dataset = BratsDataset(data, modalities_to_use, sampling_method, patch_size, transforms)
    train_loader = DataLoader(dataset=train_dataset,batch_size=16, shuffle=True, num_workers=1)


    for idx, b_data, b_labels in train_loader:
        print(b_data)


