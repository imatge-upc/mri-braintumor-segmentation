import numpy as np
import pytest
from dataset.augmentations.brats_augmentations import zero_mean_unit_variance_normalization


@pytest.fixture(scope="function")
def volume():
    return np.array([[[0., 0., 0., 0., 0.],
            [1., 1., 3., 5., 3.],
            [4., 3., 10., 30., 0.]],
           [[0., 0., 0., 0., 0.],
            [1., 1., 3., 5., 3.],
            [4., 3., 10., 30., 0.]],
           [[0., 0., 0., 0., 0.],
            [1., 1., 3., 5., 3.],
            [4., 3., 10., 30., 0.]]
           ])

def test_zero_mean_unit_variance_normalization(volume):
    normalized_volume = zero_mean_unit_variance_normalization(volume)
    assert round(normalized_volume[normalized_volume != 0].mean()) == 0.0
    assert round(normalized_volume[normalized_volume != 0].std()) == 1.0