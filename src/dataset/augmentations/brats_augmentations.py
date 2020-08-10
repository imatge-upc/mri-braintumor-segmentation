from typing import Tuple
import numpy as np
import random
import torch


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



class RandomIntensityScale(object):

    def __init__(self):
        super().__init__()

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray, np.ndarray])  -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Args:
            img_and_mask[0]: data with  all channels [C, W, H, D]
            img_and_mask[1]: segmentation mask [ W, H, D]
            img_and_mask[2]:binary mas [ W, H, D]
        Returns:
            Tuple with modalities mask and binary mask

        """
        modalities, _,  mask = img_and_mask
        scale = random.uniform(0.9, 1.1)
        modalities = modalities * scale

        return modalities, img_and_mask[1], img_and_mask[2]



class RandomIntensityShift(object):

    def __init__(self):
        super().__init__()

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray,  np.ndarray])  -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Args:
            img_and_mask[0]: data with  all channels [C, W, H, D]
            img_and_mask[1]: segmentation mask [ W, H, D]
            img_and_mask[2]:binary mas [ W, H, D]
        Returns:
        """
        modalities, _, mask = img_and_mask
        assert len(modalities.shape) == 4

        for i, modality in enumerate(modalities):

            shift = random.uniform(-0.1, 0.1)
            std = np.std(modality[mask == 1])
            modalities[i, ...] = modality + std * shift

        return modalities, img_and_mask[1], mask



class RandomMirrorFlip(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray,  np.ndarray])  -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Args:
           img_and_mask[0]: data with  all channels [C, W, H, D]
            img_and_mask[1]: segmentation mask [ W, H, D]
            img_and_mask[2]:binary mas [ W, H, D]

        Returns:
            numpy array or Tensor: Randomly flipped image.
        """
        modalities, seg_mask, mask = img_and_mask
        assert len(modalities.shape) == 4
        assert len(seg_mask.shape) == 3

        if torch.rand(1) < self.p:
            modalities = np.flip(modalities, axis=[1, 2, 3])
            seg_mask = np.flip(seg_mask, axis=[0,1,2])

        return modalities, seg_mask, mask