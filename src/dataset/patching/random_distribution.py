import numpy as np
from src.dataset.patching.commons import array4d_crop, fix_crop_center_3d


def _select_random_start_in_tumor(brain_mask, patch_size):

    brain_indices = np.nonzero(brain_mask)
    start_x =  np.random.randint(0, len(brain_indices[0]))
    start_y = np.random.randint(0, len(brain_indices[1]))
    start_z = np.random.randint(0, len(brain_indices[2]))
    center_coord = (brain_indices[0][start_x], brain_indices[1][start_y], brain_indices[2][start_z])

    return fix_crop_center_3d(brain_mask, patch_size, center_coord)


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple, mask: np.ndarray):
    """
    Randomly chosen inside the tumor region
    """
    center = _select_random_start_in_tumor(mask, patch_size)
    volume_patch = array4d_crop(volume, patch_size, center)
    seg_patch = array4d_crop(labels[None], patch_size, center)[0, :, :, :]

    return volume_patch, seg_patch

