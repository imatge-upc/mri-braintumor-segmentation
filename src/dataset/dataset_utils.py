import os
from typing import Tuple, List
import numpy as np
import csv
from src.dataset.patient import Patient

def read_brats(csv_path: str) -> List:
    patients = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            patients.append(Patient(idx=row[0], center=row[3], grade=row[1], patient=row[2], patch_name=row[4],
                                    size=list(map(int, row[5].split("x"))), data_path=os.path.dirname(csv_path)))
    return patients

def get_dataset(rootdir: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.empty([0, 4], dtype='object')
    ground_truth = np.empty([0], dtype='object')
    _, dirs, _ = next(os.walk(rootdir))

    for tumor_type in dirs:
        tumor_type_path = os.path.join(rootdir, tumor_type)
        for _, dirs, _ in os.walk(tumor_type_path):
            for person in dirs:
                base_person_file = os.path.join(tumor_type_path, person, person)
                data = np.vstack((data, [
                    "{}_flair.nii.gz".format(base_person_file),
                    "{}_t1.nii.gz".format(base_person_file),
                    "{}_t2.nii.gz".format(base_person_file),
                    "{}_t1ce.nii.gz".format(base_person_file)
                ]))
                ground_truth = np.append(ground_truth, "{}_seg.nii.gz".format(base_person_file))

    return data, ground_truth


