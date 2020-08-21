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
            seg_mask = np.flip(seg_mask, axis=[0, 1, 2])

        return modalities, seg_mask, mask



class RandomRotation90(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p


    def _augment_rot90(self, sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
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
        axes = [i + 1 for i in axes]

        sample_data = np.rot90(sample_data, num_rot, axes)
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
        assert len(modalities.shape) == 4
        assert len(seg_mask.shape) == 3

        modalities, seg_mask =  self._augment_rot90(modalities, seg_mask)
        return modalities, seg_mask, mask



class GammaCorrection(object):

    def __init__(self, p=0.5, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False, retain_stats=False):
        super().__init__()
        self.p = p
        self.invert_image = invert_image
        self.gamma_range = gamma_range
        self.epsilon = epsilon
        self.per_channel = per_channel
        self.retain_stats = retain_stats


    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray,  np.ndarray])  -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        data_sample, seg_mask, mask = img_and_mask


        if self.invert_image:
            data_sample = - data_sample
        if not self.per_channel:
            if self.retain_stats:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
            if self.retain_stats:
                data_sample = data_sample - data_sample.mean() + mn
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
        else:
            for c in range(data_sample.shape[0]):
                if self.retain_stats:
                    mn = data_sample[c].mean()
                    sd = data_sample[c].std()
                if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                    gamma = np.random.uniform(self.gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
                minm = data_sample[c].min()
                rnge = data_sample[c].max() - minm
                data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + self.epsilon)), gamma) * float(
                    rnge + self.epsilon) + minm
                if self.retain_stats:
                    data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                    data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd

        if self.invert_image:
            data_sample = - data_sample

        return data_sample, seg_mask, mask



class ChannelTranslation():
    """Simulates badly aligned color channels/modalities by shifting them against each other
    Args:
        const_channel: Which color channel is constant? The others are shifted
        max_shifts (dict {'x':2, 'y':2, 'z':2}): How many pixels should be shifted for each channel?
    """

    def __init__(self, const_channel=0, max_shifts=None, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.max_shift = max_shifts
        self.const_channel = const_channel


    def augment_channel_translation(self, data, const_channel=0, max_shifts=None):
        if max_shifts is None:
            max_shifts = {'z': 2, 'y': 2, 'x': 2}

        shape = data.shape

        const_data = data[:, [const_channel]]
        trans_data = data[:, [i for i in range(shape[1]) if i != const_channel]]

        # iterate the batch dimension
        for j in range(shape[0]):

            slice = trans_data[j]

            ixs = {}
            pad = {}

            if len(shape) == 5:
                dims = ['z', 'y', 'x']
            else:
                dims = ['y', 'x']

            # iterate the image dimensions, randomly draw shifts/translations
            for i, v in enumerate(dims):
                rand_shift = np.random.choice(list(range(-max_shifts[v], max_shifts[v], 1)))

                if rand_shift > 0:
                    ixs[v] = {'lo': 0, 'hi': -rand_shift}
                    pad[v] = {'lo': rand_shift, 'hi': 0}
                else:
                    ixs[v] = {'lo': abs(rand_shift), 'hi': shape[2 + i]}
                    pad[v] = {'lo': 0, 'hi': abs(rand_shift)}

            # shift and pad so as to retain the original image shape
            if len(shape) == 5:
                slice = slice[:, ixs['z']['lo']:ixs['z']['hi'], ixs['y']['lo']:ixs['y']['hi'],
                        ixs['x']['lo']:ixs['x']['hi']]
                slice = np.pad(slice, ((0, 0), (pad['z']['lo'], pad['z']['hi']), (pad['y']['lo'], pad['y']['hi']),
                                       (pad['x']['lo'], pad['x']['hi'])),
                               mode='constant', constant_values=(0, 0))
            if len(shape) == 4:
                slice = slice[:, ixs['y']['lo']:ixs['y']['hi'], ixs['x']['lo']:ixs['x']['hi']]
                slice = np.pad(slice, ((0, 0), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])),
                               mode='constant', constant_values=(0, 0))

            trans_data[j] = slice

        data_return = np.concatenate([const_data, trans_data], axis=1)
        return data_return

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray,  np.ndarray])  -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        data_sample, seg_mask, mask = img_and_mask


        ret_val = self.augment_channel_translation(data=data_sample, const_channel=self.const_channel, max_shifts=self.max_shift)

        data = ret_val[0]

        return data, seg_mask, mask