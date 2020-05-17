import pytest

from src.dataset.visualization_utils import plot_3_view
from src.dataset.patching.random_distribution import patching
from tests.dataset.patching.common import patching_strategy


def test_assert_patch_shape():
    volume, volume_patch, seg_patch = patching_strategy(patching, (128, 128, 128))
    assert volume_patch.shape == (4, 128, 128, 128)
    assert seg_patch.shape == (128, 128, 128)


# @pytest.mark.skip
def test_visual_test():
    volume, volume_patch, seg_patch = patching_strategy(patching, (128, 128, 128))
    plot_3_view("random_flair", volume[0, :, :, :], 100, save=True)
    plot_3_view("random_patch_flair", volume_patch[0, :, :, :], 64, save=True)
    plot_3_view("random_path_seg", seg_patch, 64, save=True)