import random
import numpy as np


def array4d_center_crop(data: np.ndarray, new_shape: tuple):
    assert len(data.shape) == 4
    startx = data.shape[1] // 2 - (new_shape[0] // 2)
    starty = data.shape[2] // 2 - (new_shape[1] // 2)
    startz = data.shape[3] // 2 - (new_shape[2] // 2)

    return data[:, startx:startx + new_shape[0], starty:starty + new_shape[1], startz:startz + new_shape[2]]

def array4d_crop(data: np.ndarray, new_shape: tuple, center: tuple):
    assert len(data.shape) == 4
    return data[:,
           center[0] - new_shape[0]//2:center[0] + new_shape[0]//2,
           center[1] - new_shape[1]//2:center[1] + new_shape[1]//2,
           center[2] - new_shape[2]//2:center[2] + new_shape[2]//2]


def fix_crop_center_3d(data: np.ndarray, patch_size: tuple, center: tuple):
    def fix_center_coordinate(current_size: int, patch_size: int, center: int):
        if center + (patch_size // 2) > current_size:
            return center - (center + patch_size // 2 - current_size)
        elif center - (patch_size // 2) < 0:
            return center - (center - (patch_size // 2))
        else:
            return center

    real_center_x = fix_center_coordinate(data.shape[0], patch_size[0], center[0])
    real_center_y = fix_center_coordinate(data.shape[1], patch_size[1], center[1])
    real_center_z = fix_center_coordinate(data.shape[2], patch_size[2], center[2])
    return real_center_x, real_center_y, real_center_z

def array3d_center_crop(data: np.ndarray, new_shape: tuple):
    assert len(data.shape) == 3
    return array4d_center_crop(data[None], new_shape)[0, :, :, :]


def crop_volume_margin(volume, patch_size):
    assert len(volume.shape) == 3

    crop_shape = (volume.shape[0] - patch_size[0],
                  volume.shape[1] - patch_size[1],
                  volume.shape[2] - patch_size[2])

    return array3d_center_crop(volume, crop_shape)


def select_patch_by_label_distribution(volume, segmentation_mask, patch_size, function):

    labels = list(np.unique(segmentation_mask))
    labels = list(set(map(function, labels)))
    selected_label = random.choice(labels)

    vf = np.vectorize(function)
    segmentation_mask_new = vf(segmentation_mask)

    segmentation_mask_new = crop_volume_margin(segmentation_mask_new, patch_size)

    positions = np.argwhere(segmentation_mask_new == selected_label)
    axis_center = np.random.randint(0, len(positions))
    center_x, center_y, center_z = positions[axis_center]

    seg_patch = segmentation_mask[center_x:center_x + patch_size[0], center_y:center_y + patch_size[1], center_z:center_z + patch_size[2]]
    volume_patch = volume[:, center_x:center_x + patch_size[0], center_y:center_y + patch_size[1], center_z:center_z + patch_size[2]]

    return volume_patch, seg_patch
