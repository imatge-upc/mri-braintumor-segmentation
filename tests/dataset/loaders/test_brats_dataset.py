import time

import pytest
from src.dataset.loaders.brats_dataset import BratsDataset
from src.dataset.patient import Patient


@pytest.fixture("function")
def dataset():
    data = [Patient(idx="", center="", grade="", patient="BraTS20_Training_001", patch_name="BraTS20_Training_001",
           size=[240, 240, 155] , data_path="/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/",
                    train=True)]*10
    return data

def test_dataset_no_patch(dataset):
    brats_dataset = BratsDataset(dataset, None, (240, 240, 155), None, compute_patch=False)
    start = time.time()
    modalities, segmentation = brats_dataset.__getitem__(0)
    print("\n Time: ", time.time()-start)
    assert modalities.shape == (4, 240, 240, 155)
    assert segmentation.shape == (240, 240, 155)

def test_dataset_random_distribution(dataset):
    from src.dataset.patching import random_distribution
    brats_dataset = BratsDataset(dataset, random_distribution, (128, 128, 128), transforms=None, compute_patch=True)
    start = time.time()
    modalities, segmentation = brats_dataset.__getitem__(0)
    print("\n Time: ", time.time()-start)
    assert modalities.shape == (4, 128, 128, 128)
    assert segmentation.shape == (128, 128, 128)

def test_dataset_random_tumor_distribution(dataset):
    from src.dataset.patching import random_tumor_distribution
    brats_dataset = BratsDataset(dataset, random_tumor_distribution, (128, 128, 128), transforms=None, compute_patch=True)
    start = time.time()
    modalities, segmentation = brats_dataset.__getitem__(0)
    print("\n Time: ", time.time()-start)
    assert modalities.shape == (4, 128, 128, 128)
    assert segmentation.shape == (128, 128, 128)


def test_dataset_random_tumor_distribution_multiple_calls(dataset):
    from src.dataset.patching import random_tumor_distribution
    brats_dataset = BratsDataset(dataset, random_tumor_distribution, (128, 128, 128), transforms=None,
                                 compute_patch=True)

    start = time.time()
    for idx in range(0, 5):
        modalities, segmentation = brats_dataset.__getitem__(idx)
    print("\n Time: ", time.time() - start)
