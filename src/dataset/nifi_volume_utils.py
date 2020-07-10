from typing import Tuple

import numpy as np
import nibabel as nib
from src.dataset.augmentations.brats_augmentations import zero_mean_unit_variance_normalization


def get_one_label_volume(mask: np.ndarray, label: int) -> np.ndarray:
    selector = lambda x: x if x == label else 0
    vfunc = np.vectorize(selector)
    return vfunc(mask)


def save_nifi_volume(volume:np.ndarray, path:str):
    img = nib.Nifti1Image(volume, np.eye(4))
    img.header.get_xyzt_units()
    img.to_filename(path)

def save_segmask_as_nifi_volume(seg_mask: np.ndarray, volume_path: str , path:str):
    aff_func = nib.load(volume_path).affine
    img = nib.Nifti1Image(seg_mask, aff_func)
    img.to_filename(path)

def load_nifi_volume(filepath: str, normalize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    nib_data = nib.load(filepath)
    img = nib_data.get_fdata()
    nib_data.uncache()
    if normalize:
        img = zero_mean_unit_variance_normalization(img)
    return img, nib_data

