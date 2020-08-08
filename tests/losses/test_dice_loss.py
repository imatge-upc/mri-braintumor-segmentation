import os
import pytest
import torch
from mock import MagicMock
from src.dataset.utils import nifi_volume
from src.losses.ce_dice_loss import CrossEntropyDiceLoss3D
from src.losses import dice_loss
from src.models.vnet import vnet
from torch import nn


@pytest.fixture(scope="function")
def volume():
    patient = "BraTS20_Training_001_p0_64x64x64"
    gen_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/random_tumor_distribution/"
    volume_path  = os.path.join(gen_path, patient, f"{patient}_seg.nii.gz")
    return nifi_volume.load_nifi_volume(volume_path, normalize=False)

class Identity(nn.Module):

    def forward(self, input):
        return input

def test_dice_loss(volume):

    volume[volume == 4] = 3
    classes = 4

    my_loss = dice_loss.DiceLoss(classes=classes, weight=None, sigmoid_normalization=True, eval_regions=True)


    seg_mask = torch.from_numpy(volume.astype(int))

    target = seg_mask.unsqueeze(0).to("cpu")

    input = dice_loss.expand_as_one_hot(target.long(), classes)

    my_loss.normalization = Identity()
    loss, score, _ = my_loss(input, target)

    assert loss == 0
    assert score == 1