import csv
import sys
import os
import numpy as np
from src.dataset import nifi_volume_utils as nifi_utils
from src.config import BratsConfiguration
from src.dataset import dataset_utils
from src.dataset import brats_labels
from src.metrics import evaluation_metrics as eval
from tqdm import tqdm


def compute(volume_pred, volume_gt):
    tp, fp, tn, fn = eval.get_confusion_matrix(volume_pred, volume_gt)
    dc = eval.dice(tp, fp, fn)

    hd = eval.hausdorff(volume_pred, volume_gt)
    recall  = eval.recall(tp, fn)
    precision = eval.precision(tn, fp)
    acc = eval.accuracy(tp, fp, tn, fn)
    f1 = eval.fscore(tp, fp, tn ,fn)

    return dc, hd, recall, precision, acc, f1, (tp, fp, tn, fn)


if __name__ == "__main__":
    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()

    data_train, data_test = dataset_utils.read_brats(dataset_config.get("train_csv"))
    data = data_test

    with open(f"results_test.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["subject_ID", "Grade", "Center", "Size",
                         "Dice WT", "HD WT", "Recall WT", "Precision WT", "Acc WT", "F1 WT", "CONF WT",
                         "Dice TC", "HD TC", "Recall TC", "Precision TC", "Acc TC", "F1 TC", "CONF TC",
                         "Dice ET", "HD ET", "Recall ET", "Precision ET", "Acc ET", "F1 ET", "CONF ET"
                         ])

        for patient in tqdm(data, total=len(data)):

            patient_data = []
            gt_path = os.path.join(patient.data_path, patient.patient, f"{patient.seg}")
            prediction_path  = os.path.join(patient.data_path, patient.patient, f"{patient.patient}_prediction.nii.gz")
            if not os.path.exists(prediction_path):
                print(f"{prediction_path} not found")
                continue

            patient_data.extend([patient.patient, patient.grade, patient.center, patient.size])

            volume_gt_all = nifi_utils.load_nifi_volume(gt_path)
            volume_pred_all = nifi_utils.load_nifi_volume(prediction_path)


            for typee in ["wt", "tc", "et"]:
                compute_metrics = True
                if typee == "wt":
                    volume_gt = brats_labels.get_wt(volume_gt_all)
                    volume_pred = brats_labels.get_wt(volume_pred_all)

                elif typee == "tc":
                    volume_gt = brats_labels.get_tc(volume_gt_all)
                    volume_pred = brats_labels.get_tc(volume_pred_all)

                elif typee == "et":
                    volume_gt = brats_labels.get_et(volume_gt_all)
                    volume_pred = brats_labels.get_et(volume_pred_all)
                    if len(np.unique(volume_gt)) == 1 and np.unique(volume_gt)[0] == 0:
                        print("there is no enchancing tumor region in the ground truth")

                dc, hd, recall, precision, acc, f1, conf_matrix = compute(volume_pred, volume_gt)
                patient_data.extend([dc, hd, recall, precision, acc, f1, conf_matrix])


            writer.writerow(patient_data)
