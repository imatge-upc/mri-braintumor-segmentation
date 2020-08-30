from src.dataset.patching.binary_distribution import patching
from tests.dataset.patching.common import patching_strategy, save
from src.dataset.utils.visualization import plot_3_view


def test_assert_patch_shape():
    volume, volume_patch, seg_patch = patching_strategy(patching, (96, 96, 96))
    assert volume_patch.shape == (4, 80, 80, 80)
    assert seg_patch.shape == (80, 80, 80)


def test_visual_test():
    volume, volume_patch, seg_patch = patching_strategy(patching, (80, 80, 80))
    # plot_3_view("binary_flair", volume[0, :, :, :], 100, save=save)
    plot_3_view("binary_patch_flair", volume_patch[0, :, :, :], 40, save=save)
    plot_3_view("binary_path_seg", seg_patch[:, :, :], 40, save=save)