import os
from typing import Tuple

from skimage.morphology import remove_small_objects
from src.compute_metric_results import compute_wt_tc_et

from src.dataset.utils.nifi_volume import load_nifi_volume_return_nib, save_segmask_as_nifi_volume


def load_volume(path) -> Tuple:
    return load_nifi_volume_return_nib(path, normalize=False)


def remove_small_elements(segmentation_mask, min_size=1000):

    pred_mask = segmentation_mask > 0
    mask = remove_small_objects(pred_mask, min_size=min_size)
    clean_segmentation = segmentation * mask
    return clean_segmentation




def compute_metrics(ground_truth_path, subject, segmentation, clean_segmentation):
    gt_path = os.path.join(ground_truth_path, subject, f"{subject}_seg.nii.gz")
    data_path = os.path.join(ground_truth_path, subject, f"{subject}_flair.nii.gz")

    volume_gt, _ = load_volume(gt_path)
    volume, _ = load_volume(data_path)

    metrics_before = compute_wt_tc_et(segmentation, volume_gt, volume)
    metrics_after = compute_wt_tc_et(clean_segmentation, volume_gt, volume)

    print(f"{subject} - Before {metrics_before}")
    print(f"{subject} - After  {metrics_after}")


if __name__ == "__main__":

    ground_truth_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch"

    model_path =  "results/checkpoints/segmentation_task/"
    output_dir = os.path.join(model_path, "segmentation_task_clean")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, filename in enumerate(os.listdir(model_path)):
        if "BraTS20" not in filename:
            continue

        subject = filename.split(".")[0]
        output_path = os.path.join(output_dir, f"{subject}.nii.gz")
        prediction_path = os.path.join(model_path, f"{subject}.nii.gz")


        segmentation, segmentation_nib = load_volume(prediction_path)

        clean_segmentation = remove_small_elements(segmentation)
        save_segmask_as_nifi_volume(clean_segmentation, segmentation_nib.affine, output_path)

        compute_metrics(ground_truth_path, subject, segmentation, clean_segmentation)

        if i > 50:
            break