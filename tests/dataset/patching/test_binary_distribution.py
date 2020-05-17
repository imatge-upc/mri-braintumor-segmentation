import pytest

from src.dataset.patching.binary_distribution import patching
from tests.dataset.patching.common import patching_strategy
from src.dataset.visualization_utils import plot_3_view


def test_assert_patch_shape():
    volume, volume_patch, seg_patch = patching_strategy(patching, (80, 80, 80))
    assert volume_patch.shape == (4, 80, 80, 80)
    assert seg_patch.shape == (80, 80, 80)


# @pytest.mark.skip
def test_visual_test():
    volume, volume_patch, seg_patch = patching_strategy(patching, (80, 80, 80))
    plot_3_view("binary_flair", volume[0, :, :, :], 100, save=True)
    plot_3_view("binary_patch_flair", volume_patch[0, :, :, :], 40, save=True)
    plot_3_view("binary_path_seg", seg_patch, 40, save=True)