import numpy as np
import pytest
from src.dataset.augmentations.spatial_augmentations import RandomRotation90
from src.dataset.utils.visualization import plot_3_view
from tests.dataset.patching.common import load_patient, get_brain_mask


@pytest.fixture(scope="function")
def volume():
    return np.array([[[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.]],
                     [[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.]],
                     [[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.]]
                     ])


@pytest.fixture(scope="function")
def patient():
    brain, seg = load_patient()
    return brain, seg, get_brain_mask()


def test_random_rotation_90(volume):
    rot = RandomRotation90(p=1)
    img = np.expand_dims(volume, axis=0)
    img, seg, brain_mask = rot.__call__(img_and_mask=(img, volume, volume))
    assert img.shape == img.shape


def test_random_rotation_90_real_patient(patient):
    volume, seg, brain_mask = patient
    rot = RandomRotation90(p=1)
    rot_volume, rot_seg, _ = rot.__call__(img_and_mask=(volume, seg, brain_mask))

    plot_3_view("rotated_volume", rot_volume[0, :, :, :], 100, save=True)
    plot_3_view("rotated_seg", rot_seg[:, :, :], 100, save=True)
    plot_3_view("volume", volume[0, :, :, :], 100, save=True)
    plot_3_view("seg", seg[:, :, :], 100, save=True)