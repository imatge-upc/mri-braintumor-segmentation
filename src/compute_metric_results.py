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


def compute(volume_pred, volume_gt, compute_hd=True):
    tp, fp, tn, fn = eval.get_confusion_matrix(volume_pred, volume_gt)
    dc = eval.dice(tp, fp, fn)

    if not compute_hd:
        return dc, "", "", "", "", ""

    hd = eval.hausdorff(volume_pred, volume_gt)
    recall  = eval.sensitivity(tp, fn)
    precision = eval.specificity(tn, fp)
    acc = eval.accuracy(tp, fp, tn, fn)
    f1 = eval.fscore(tp, fp, tn , fn)

    return dc, hd, recall, precision, acc, f1


if __name__ == "__main__":
    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()

    data_train, data_test = dataset_utils.read_brats(dataset_config.get("train_csv"))

    with open(f"results_train.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["subject_ID", "Grade", "Center", "Size",
                         "Dice WT", "Dice TC", "Dice ET",
                         "HD WT", "HD TC", "HD ET",
                         "Recall WT",  "Recall TC", "Recall ET",
                         "Precision WT", "Precision TC", "Precision ET",
                         "Acc WT", "Acc TC", "Acc ET",
                         "F1 WT", "F1 TC", "F1 ET"
                         ])

        for test_patient in tqdm(data_train, total=len(data_test)):

            patient_data = []
            gt_path = os.path.join(test_patient.data_path, test_patient.patient, f"{test_patient.seg}")
            prediction_path  = os.path.join(test_patient.data_path, test_patient.patient, f"{test_patient.patient}_prediction.nii.gz")
            if not os.path.exists(prediction_path):
                print(f"{prediction_path} not found")
                continue

            patient_data.extend([test_patient.patient, test_patient.grade, test_patient.center, test_patient.size])

            volume_gt = nifi_utils.load_nifi_volume(gt_path)
            volume_pred = nifi_utils.load_nifi_volume(prediction_path)

            # whole tumor
            volume_gt_wt = brats_labels.get_wt(volume_gt)
            volume_pred_wt = brats_labels.get_wt(volume_pred)
            dc_wt, hd_wt, recall_wt, precision_wt, acc_wt, f1_wt = compute(volume_pred_wt, volume_gt_wt)

            # tumor core
            volume_gt_tc= brats_labels.get_tc(volume_gt)
            volume_pred_tc = brats_labels.get_tc(volume_pred)
            dc_tc, hd_tc, recall_tc, precision_tc, acc_tc, f1_tc = compute(volume_pred_tc, volume_gt_tc)

            # enhancing tumor
            volume_gt_et = brats_labels.get_et(volume_gt)
            volume_pred_et = brats_labels.get_et(volume_pred)
            if len(np.unique(volume_gt_et)) == 1 and np.unique(volume_gt_et)[0] ==0 :
                print("there is no enchancing tumor region in the ground truth")
                dc_et, hd_et, recall_et, precision_et, acc_et, f1_et = compute(volume_pred_et, volume_gt_et, False)

            else:
                dc_et, hd_et, recall_et, precision_et, acc_et, f1_et = compute(volume_pred_et, volume_gt_et)

            patient_data.extend([dc_wt, dc_tc, dc_et,
                                 hd_wt, hd_tc, hd_et,
                                 recall_wt, recall_tc, recall_et,
                                 precision_wt, precision_tc, precision_et,
                                 acc_wt, acc_tc, acc_et,
                                 f1_wt, f1_tc, f1_et])

            writer.writerow(patient_data)
