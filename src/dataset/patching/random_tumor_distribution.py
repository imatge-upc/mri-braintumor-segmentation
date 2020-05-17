import numpy as np
from src.dataset.patching.commons import array3d_center_crop


def _select_random_center_in_tumor(segmentation_mask, patch_size):

    crop_shape = (segmentation_mask.shape[0]-patch_size[0],
                  segmentation_mask.shape[1]-patch_size[1],
                  segmentation_mask.shape[2]-patch_size[2])

    cropped_mask = array3d_center_crop(segmentation_mask, crop_shape)

    tumor_region_indices = np.nonzero(cropped_mask)
    axis_center = np.random.randint(0, len(tumor_region_indices[0]))

    return tumor_region_indices[0][axis_center], tumor_region_indices[1][axis_center], tumor_region_indices[2][axis_center]


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple):
    """
    Randomly chosen inside the tumor region
    """
    center_x, center_y, center_z = _select_random_center_in_tumor(labels, patch_size)
    volume_patch = volume[:, center_x:center_x+patch_size[0], center_y:center_y+patch_size[1], center_z:center_z+patch_size[2]]
    seg_patch = labels[center_x:center_x + patch_size[0], center_y:center_y + patch_size[1], center_z:center_z + patch_size[2]]

    return volume_patch, seg_patch
