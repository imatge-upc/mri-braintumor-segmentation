import os
import numpy as np
from src.dataset.utils.nifi_volume import load_nifi_volume

class Patient:
    def __init__(self, idx: str, center: str, grade: str, patient: str, patch_name: str,
                 size: list, data_path: str, train: bool):

        self.grade = grade
        self.center = center
        self.id = idx
        self.patient = patient
        self.data_path = data_path
        self.size = size
        self.patch_name = patch_name
        self.train = train
        self.t1ce = f"{self.patch_name}_t1ce.nii.gz"
        self.t1 = f"{self.patch_name}_t1.nii.gz"
        self.t2 = f"{self.patch_name}_t2.nii.gz"
        self.flair = f"{self.patch_name}_flair.nii.gz"
        self.seg = f"{self.patch_name}_seg.nii.gz"


    def load_mri_volumes(self) -> np.ndarray:
        patient_path = os.path.join(self.data_path, self.patch_name)

        flair, _ = load_nifi_volume(os.path.join(patient_path, self.flair), True)
        t1, _ = load_nifi_volume(os.path.join(patient_path, self.t1), True)
        t2, _ = load_nifi_volume(os.path.join(patient_path, self.t2), True)
        t1_ce, _ = load_nifi_volume(os.path.join(patient_path, self.t1ce), True)
        modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

        return modalities

    def load_gt_mask(self) -> np.ndarray:
        patient_path = os.path.join(self.data_path, self.patch_name)
        volume, _ = load_nifi_volume(os.path.join(patient_path, self.seg), normalize=False)
        return volume
