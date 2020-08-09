import os
from typing import List, Tuple
import numpy as np
import csv
from src.dataset.patient import Patient

def read_brats(csv_path: str) -> Tuple[List, List]:
    patients_test = []
    patients_train = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            if row[6] == "train":
                patients_train.append(Patient(idx=row[0], center=row[3], grade=row[1], patient=row[2], patch_name=row[4],
                                              size=list(map(int, row[5].split("x"))), data_path=os.path.dirname(csv_path),
                                              train=True))
            else:
                patients_test.append(Patient(idx=row[0], center=row[3], grade=row[1], patient=row[2], patch_name=row[4],
                                              size=list(map(int, row[5].split("x"))), data_path=os.path.dirname(csv_path),
                                              train=False))
    return patients_train, patients_test


def create_roi_mask(data: np.ndarray) -> np.ndarray:
    # filter values bigger than 0
    brain_mask = np.zeros(data.shape, np.float)
    brain_mask[data > 0] = 1
    return brain_mask
