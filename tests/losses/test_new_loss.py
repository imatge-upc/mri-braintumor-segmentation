import os
import pytest
import torch
from src.dataset.utils import nifi_volume
from src.losses import new_losses, utils
from torch import nn


@pytest.fixture(scope="function")
def gt_mask():
    patient = "BraTS20_Training_001_p0_64x64x64"
    gen_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/random_tumor_distribution/"
    volume_path  = os.path.join(gen_path, patient, f"{patient}_seg.nii.gz")
    return nifi_volume.load_nifi_volume(volume_path, normalize=False)

@pytest.fixture(scope="function")
def volume_flair():
    patient = "BraTS20_Training_001_p0_64x64x64"
    gen_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/random_tumor_distribution/"
    volume_path  = os.path.join(gen_path, patient, f"{patient}_flair.nii.gz")
    return nifi_volume.load_nifi_volume(volume_path, normalize=True)


class Identity(nn.Module):

    def forward(self, input):
        return input

def test_dice_loss(gt_mask):

    gt_mask[gt_mask == 4] = 3

    my_loss = new_losses.GeneralizedDiceLoss()

    seg_mask = torch.from_numpy(gt_mask.astype(int))

    target = seg_mask.unsqueeze(0).to("cpu")
    target = utils.expand_as_one_hot(target, num_classes=4)

    my_loss.normalization = Identity()

    loss, score = my_loss(target, target)

    assert round(loss.item()) == 0
    assert round(score.item()) == 1


def test_dice_los_real_results(gt_mask, volume_flair):
    from src.models.vnet import vnet

    gt_mask[gt_mask == 4] = 3

    network = vnet.VNet(elu=True, in_channels=1, classes=4, init_features_maps=16)
    network.to("cpu")

    my_loss = new_losses.GeneralizedDiceLoss()

    network.train()
    volume_flair = torch.from_numpy(volume_flair).unsqueeze(0).unsqueeze(0)
    outputs, scores = network(volume_flair.float())

    seg_mask = torch.from_numpy(gt_mask.astype(int))

    target = seg_mask.unsqueeze(0).to("cpu")
    target = utils.expand_as_one_hot(target, num_classes=4)


    loss, score = my_loss(outputs, target)

    assert round(loss.item()) == 1
    assert round(score.item()) == 0