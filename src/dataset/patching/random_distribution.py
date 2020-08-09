import numpy as np


# Implementation from the actual paper: no new net https://github.com/MIC-DKFZ/BraTS2017/blob/master/dataset.py

def _select_random_center_axis(volume_axis_value, crop_axis_value, axis="x"):

    if crop_axis_value < volume_axis_value:
        axis_center = np.random.randint(0, volume_axis_value - crop_axis_value)
    elif crop_axis_value == volume_axis_value:
        axis_center = 0
    else:
        raise ValueError(f"Axis {axis} crop size must be smaller or equal to the images {axis} dimension")

    return axis_center


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple,  mask: np.ndarray):
    """
    Random sample patches. Patch size should be big
    """
    assert type(patch_size) in (tuple, list)
    assert len(patch_size) == (len(volume.shape) - 1)  # one if we only add the modalities, 2 if we also add the batch size

    center_x = _select_random_center_axis(volume.shape[1], patch_size[0], "x")
    center_y = _select_random_center_axis(volume.shape[2], patch_size[1], "y")
    center_z = _select_random_center_axis(volume.shape[3], patch_size[2], "z")

    volume_patch = volume[:, center_x:center_x+patch_size[0], center_y:center_y+patch_size[1], center_z:center_z + patch_size[2]]
    seg_patch = labels[center_x:center_x + patch_size[0], center_y:center_y + patch_size[1], center_z:center_z + patch_size[2]]

    return volume_patch, seg_patch
