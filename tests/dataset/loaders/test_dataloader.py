import time

import pytest
from src.dataset.loaders.brats_dataset import BratsDataset
from src.dataset.patient import Patient
from torch.utils.data import DataLoader


@pytest.fixture("function")
def dataset():
    data = [Patient(idx="", center="", grade="", patient="BraTS20_Training_001", patch_name="BraTS20_Training_001",
           size=[240, 240, 155] , data_path="/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/",
                    train=True)] * 10
    return data


def test_dataset_no_patch(dataset):
    bs = 5
    brats_dataset = BratsDataset(dataset, None, (240, 240, 155), transforms=None, compute_patch=False)
    loader = DataLoader(dataset=brats_dataset, batch_size=bs, shuffle=True, num_workers=1)
    start = time.time()
    modalities, segmentation = next(iter(loader))
    print("\n Time: ", time.time()-start)
    assert modalities.shape == (bs, 4, 240, 240, 155)
    assert segmentation.shape == (bs, 240, 240, 155)

def test_dataset_random_distribution(dataset):
    bs = 5
    from src.dataset.patching import random_distribution
    brats_dataset = BratsDataset(dataset, random_distribution, (128, 128, 128), transforms=None, compute_patch=True)

    loader = DataLoader(dataset=brats_dataset, batch_size=bs, shuffle=True, num_workers=1)
    start = time.time()
    modalities, segmentation = next(iter(loader))
    print("\n Time: ", time.time()-start)

    assert modalities.shape == (bs, 4, 128, 128, 128)
    assert segmentation.shape == (bs, 128, 128, 128)

def test_dataset_random_tumor_distribution(dataset):
    bs = 2
    from src.dataset.patching import random_tumor_distribution
    brats_dataset = BratsDataset(dataset, random_tumor_distribution, (128, 128, 128), transforms=None, compute_patch=True)
    loader = DataLoader(dataset=brats_dataset, batch_size=bs, shuffle=True, num_workers=1)

    start = time.time()
    modalities, segmentation = next(iter(loader))
    print("\n Time: ", time.time()-start)

    assert modalities.shape == (bs, 4, 128, 128, 128)
    assert segmentation.shape == (bs, 128, 128, 128)
