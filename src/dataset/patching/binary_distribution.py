import numpy as np
import random

from src.dataset.patching.commons import select_patch_by_label_distribution


def select_label_with_equal_prop(labels):
    return random.choice(labels)


def patching(volume: np.ndarray, labels: np.ndarray, patch_size: tuple, mask: np.ndarray):
    """

    50% probability of center being  healthy tissue - 50% to be centered in tumor regions
    Sampling technique from:
    'Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation'
    """
    binary_function = lambda label: 0 if label == 0 else 1

    volume_patch, seg_patch = select_patch_by_label_distribution(volume, labels, patch_size, binary_function, mask)
    return volume_patch, seg_patch
