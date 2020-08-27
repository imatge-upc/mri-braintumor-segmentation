from typing import Tuple
import numpy as np
import torch


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

        if torch.rand(1) < self.p:
            modalities = np.flip(modalities, axis=[1, 2, 3])
            if seg_mask is not None:
                seg_mask = np.flip(seg_mask, axis=[0, 1, 2])

        return modalities, seg_mask, mask


class RandomRotation90(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    @staticmethod
    def _augment_rot90(sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
        """
        :param sample_data:
        :param sample_seg:
        :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
        :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
        :return:
        """
        num_rot = np.random.choice(num_rot)
        axes = np.random.choice(axes, size=2, replace=False)
        axes.sort()

        axes_data = [i + 1 for i in axes]
        sample_data = np.rot90(sample_data, num_rot, axes_data)
        if sample_seg is not None:
            sample_seg = np.rot90(sample_seg, num_rot, axes)
        return sample_data, sample_seg

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
        modalities, seg_mask = self._augment_rot90(modalities, seg_mask)
        return modalities, seg_mask, mask
