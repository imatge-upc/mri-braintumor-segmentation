import numpy as np


def zero_mean_unit_variance_normalization(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize a target image by subtracting the mean of the brain region and dividing by the standard deviation
    :return: normalized volume: with 0-mean and unit-std for non-zero voxels only!
    """
    non_zero = data[data > 0.0]
    mean = non_zero.mean()
    std = non_zero.std() + epsilon
    out = (data - mean) / std
    out[data == 0] = 0
    return out