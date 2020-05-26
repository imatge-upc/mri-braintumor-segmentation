import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import io_utils
from src.dataset.augmentations.brats_augmentations import zero_mean_unit_variance_normalization



class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1ce_idx = 0, 1, 2, 3

    def __init__(self, data: np.ndarray, ground_truth: np.ndarray, modalities_to_use: dict, sampling_method,
                 patch_size: tuple, transforms: transforms):
        """

        :param data:
        :param ground_truth:
        :param modalities_to_use:
        :param sampling_method: patching method
        :param patch_size:
        :param transforms:
        """

        self.dataset_data = data
        self.dataset_segmentation = ground_truth
        self.modalities_to_use = modalities_to_use
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        flair = self._load_volume_modality(idx, BratsDataset.flair_idx)
        t1 = self._load_volume_modality(idx, BratsDataset.t1_idx)
        t2 = self._load_volume_modality(idx, BratsDataset.t2_idx)
        t1_ce = self._load_volume_modality(idx, BratsDataset.t1_idx)
        modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

        segmentation_mask = self._load_volume_gt(idx)
        segmentation_mask = self.convert_from_labels(segmentation_mask)

        patch_modality, patch_segmentation = self.sampling_method.patching(modalities, segmentation_mask, self.patch_size)
        return idx, patch_modality, patch_segmentation


    def _load_volume_modality(self, idx: int, modality: int, normalize: bool=True):
        if modality in self.modalities_to_use.keys() and self.modalities_to_use[modality]:
            volume = io_utils.load_nifi_volume(self.dataset_data[idx, modality])
            if normalize:
                volume = zero_mean_unit_variance_normalization(volume)
            return volume
        else:
            return None

    def _load_volume_gt(self, idx: int) -> np.ndarray:
        return io_utils.load_nifi_volume(self.dataset_segmentation[idx])


    def get_patient_info(self, idx):
        data= self.dataset_data[idx]
        seg = self.dataset_segmentation[idx]
        return {"name": seg.split("/")[-2], "volumes":data.tolist() , "segmentation": seg}


    def convert_from_labels(self, segmentation_map):
        segmentation_map[segmentation_map == 4] = 3
        return segmentation_map

    def convert_to_labels(self, segmentation_map):
        segmentation_map[segmentation_map == 3] = 4
        return segmentation_map