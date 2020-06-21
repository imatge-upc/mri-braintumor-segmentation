


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
