import os
import numpy as np
from src.dataset.augmentations.data_normalization import zero_mean_unit_variance_normalization


def save_patch(patch: np.ndarray, path: str):
    _, extension = os.path.splitext(path)
    assert extension == ".npz", "File extension must be numpy's compressed format"
    np.savez_compressed(path, patch)

def load_patch(path: str, normalize=True) -> np.ndarray:
    img =  np.load(path)
    if normalize:
        img = zero_mean_unit_variance_normalization(img)
    return img
