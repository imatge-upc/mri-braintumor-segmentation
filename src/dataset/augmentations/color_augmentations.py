from typing import Tuple
import numpy as np
import random
import torch


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


class RandomGaussianNoise(object):

    def __init__(self, p=0.5, noise_variance=(0, 0.5)):
        super().__init__()
        self.p = p
        self.noise_variance = noise_variance

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray,  np.ndarray]) -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Args:
        Returns:
        """
        img, _, mask = img_and_mask
        noised_image = img
        mask = np.stack([mask]*4)

        if torch.rand(1) < self.p:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = random.uniform(self.noise_variance[0], self.noise_variance[1])

            noised_image =img + np.random.normal(0.0, variance, size=img.shape)
            noised_image[mask == 0] = img[mask == 0]

        return noised_image, img_and_mask[1], mask
