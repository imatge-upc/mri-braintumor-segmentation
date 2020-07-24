import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import brats_labels
from src.dataset.utils import nifi_volume as nifi_utils



class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1ce_idx = 0, 1, 2, 3

    def __init__(self, data: list, sampling_method, patch_size: tuple, compute_patch: bool=False):
        """

        :param data:
        :param ground_truth:
        :param modalities_to_use:
        :param sampling_method: patching method
        :param patch_size:
        """
        self.data = data
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.compute_patch = compute_patch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        modalities = self.data[idx].load_mri_volumes(normalize=True)
        segmentation_mask = self.data[idx].load_gt_mask()

        segmentation_mask = brats_labels.convert_from_brats_labels(segmentation_mask)

        if self.compute_patch:
            modalities, segmentation_mask = self.sampling_method.patching(modalities, segmentation_mask, self.patch_size)


        modalities = torch.from_numpy(modalities.astype(float))
        segmentation_mask = torch.from_numpy(segmentation_mask.astype(int))

        return modalities, segmentation_mask


    def _load_volume_modality(self, modality_path: str):
        volume = nifi_utils.load_nifi_volume(modality_path, True)
        return  volume

    def _load_volume_gt(self, seg_mask: str) -> np.ndarray:
        segmentation = nifi_utils.load_nifi_volume(seg_mask, normalize=False)
        return segmentation

    def get_patient_info(self, idx):
        return {attr[0]: attr[1] for attr in vars(self.data[idx]).items()}
