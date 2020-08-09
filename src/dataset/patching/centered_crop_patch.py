from typing import Tuple

import numpy as np
from src.dataset.patching.commons import array4d_center_crop, array3d_center_crop


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centered crop patch. Just one patch per patient
    """
    volume_patch = array4d_center_crop(volume, patch_size)
    seg_patch =  array3d_center_crop(labels, patch_size)

    return volume_patch, seg_patch
