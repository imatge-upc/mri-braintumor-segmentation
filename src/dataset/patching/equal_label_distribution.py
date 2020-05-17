import numpy as np
from typing import Tuple
from src.dataset.patching.commons import select_patch_by_label_distribution


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Patches with equal probability from each label
    """
    identity_function = lambda label: label
    volume_patch, seg_patch = select_patch_by_label_distribution(volume, labels, patch_size, identity_function)
    return volume_patch, seg_patch