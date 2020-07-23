import os
import numpy as np
import nibabel as nib
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

        extension = "nii.gz"
        self.t1ce = f"{self.patch_name}_t1ce.{extension}"
        self.t1 = f"{self.patch_name}_t1.{extension}"
        self.t2 = f"{self.patch_name}_t2.nii.gz"
        self.flair = f"{self.patch_name}_flair.{extension}"
        self.seg = f"{self.patch_name}_seg.{extension}"


    def load_mri_volumes(self, normalize) -> np.ndarray:
        patient_path = os.path.join(self.data_path, self.patch_name)

        flair = load_nifi_volume(os.path.join(patient_path, self.flair), normalize)
        t1 = load_nifi_volume(os.path.join(patient_path, self.t1), normalize)
        t2 = load_nifi_volume(os.path.join(patient_path, self.t2), normalize)
        t1_ce = load_nifi_volume(os.path.join(patient_path, self.t1ce), normalize)
        modalities = np.stack((flair, t1, t2, t1_ce))

        return modalities

    def load_gt_mask(self) -> np.ndarray:
        patient_path = os.path.join(self.data_path, self.patch_name)
        volume = load_nifi_volume(os.path.join(patient_path, self.seg), normalize=False)
        return volume

    def get_affine(self):
        patient_path = os.path.join(self.data_path, self.patch_name, self.flair)
        return nib.load(patient_path).affine
