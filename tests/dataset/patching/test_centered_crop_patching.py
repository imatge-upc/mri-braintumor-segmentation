from src.dataset.patching.centered_crop_patch import patching
from tests.dataset.patching.common import patching_strategy, save
from src.dataset.visualization_utils import plot_3_view


def test_assert_patch_shape():
    volume, volume_patch, seg_patch = patching_strategy(patching, (160, 192, 128))
    assert volume_patch.shape == (4, 160, 192, 128)
    assert seg_patch.shape == (160, 192, 128)


def test_visual_test():
    volume, volume_patch, seg_patch = patching_strategy(patching, (160, 192, 128))
    plot_3_view("center_flair", volume[0, :, :, :], 100, save=save)
    plot_3_view("center_patch_flair", volume_patch[0, :, :, :], 100, save=save)
    plot_3_view("center_path_seg", seg_patch, 100, save=save)