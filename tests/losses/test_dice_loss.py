import os
import numpy as np
import pytest
import torch
from src.dataset.utils import nifi_volume
from src.losses.ce_dice_loss import CrossEntropyDiceLoss3D
from src.losses.dice_loss_eval_regions import DiceLoss


@pytest.fixture(scope="function")
def volume():
    patient = "BraTS20_Training_001"
    gen_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/"
    volume_path  = os.path.join(gen_path, patient, f"{patient}_seg.nii.gz")
    return nifi_volume.load_nifi_volume(volume_path, normalize=False)


def test_dice_loss(volume):
    volume = volume[:, :, :128]
    volume[volume == 4] = 3
    classes = 4

    my_loss = DiceLoss(classes=classes, weight=None, sigmoid_normalization=False)


    seg_mask = torch.from_numpy(volume.astype(int))


    target = seg_mask.unsqueeze(0).to("cpu")
    input = torch.nn.functional.one_hot(target.long(), classes)


    loss, score = my_loss(input, target)
    print(loss)
    print(score)


def test_combined_ce_dice(volume):
    volume = volume[:, :, :128]
    volume[volume == 4] = 3
    classes = 4

    my_loss = CrossEntropyDiceLoss3D(classes=classes, weight=None)

    seg_mask = torch.from_numpy(volume.astype(int))

    target = seg_mask.unsqueeze(0).to("cpu")
    input = torch.nn.functional.one_hot(target.long(), classes)

    total_loss, dice_loss, ce_loss, dice_score = my_loss(input.float(), target)
    print("\n Dice Loss: ", dice_loss)
    print("\n Dice Score: ", dice_score)
    print("\n CE Loss: ", ce_loss)
    print("\n Total Loss: ", total_loss)