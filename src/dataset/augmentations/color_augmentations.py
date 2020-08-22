from typing import Tuple
import numpy as np
import random



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


