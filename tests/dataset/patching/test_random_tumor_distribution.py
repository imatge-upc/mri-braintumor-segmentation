import pytest

from src.dataset.visualization_utils import plot_3_view
from src.dataset.patching.random_tumor_distribution import patching
from tests.dataset.patching.common import patching_strategy, plot


def test_assert_patch_shape():
    volume, volume_patch, seg_patch = patching_strategy(patching, (32, 32, 16))
    assert volume_patch.shape == (4, 32, 32, 16)
    assert seg_patch.shape == (32, 32, 16)


# @pytest.mark.skip
def test_visual_test():
    volume, volume_patch, seg_patch = patching_strategy(patching, (32, 32, 16))
    plot_3_view("random_tumor_flair", volume[0, :, :, :], 100, save=True)
    plot_3_view("random_tumor_patch_flair", volume_patch[0, :, :, :], 10, save=True)
    plot_3_view("random_tumor_path_seg", seg_patch, 10, save=True)
