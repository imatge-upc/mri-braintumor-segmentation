import pytest
import os
import numpy as np
from src.dataset.utils import nifi_volume as nifi_utils
from src.dataset import brats_labels
from src.dataset.utils.visualization import plot_3_view


@pytest.fixture(scope="function")
def volume():
    patient = "BraTS20_Training_001"
    gen_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch/"
    volume_path  = os.path.join(gen_path, patient, f"{patient}_seg.nii.gz")
    volume = nifi_utils.load_nifi_volume(volume_path)
    return volume


def test_get_ncr_net(volume):
    just_ncr_net = brats_labels.get_ncr_net(volume.copy())

    plot_3_view("ncr_net", just_ncr_net[:, :, :], 100, save=True)
    plot_3_view("whole", volume[:, :, :], 100, save=True)

    assert [0, 1] == list(np.unique(just_ncr_net))
