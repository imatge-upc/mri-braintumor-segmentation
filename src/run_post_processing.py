import os
from typing import Tuple

from src.compute_metric_results import compute_wt_tc_et
from src.dataset import brats_labels
from src.dataset.utils.nifi_volume import load_nifi_volume_return_nib, save_segmask_as_nifi_volume
from src.post_processing import post_process


def load_volume(path) -> Tuple:
    return load_nifi_volume_return_nib(path, normalize=False)


def compute_metrics(ground_truth_path, subject, clean_segmentation):
    gt_path = os.path.join(ground_truth_path, subject, f"{subject}_seg.nii.gz")
    data_path = os.path.join(ground_truth_path, subject, f"{subject}_flair.nii.gz")

    volume_gt, _ = load_volume(gt_path)
    volume, _ = load_volume(data_path)

    metrics_after = compute_wt_tc_et(clean_segmentation, volume_gt, volume)

    print(f"{subject},After  {metrics_after}")


if __name__ == "__main__":
    setx = "train"
    th = 1
    model_id = "model_1598640035"
    model_path = f"/mnt/gpid07/users/laura.mora/results/checkpoints/{model_id}/"

    input_dir = os.path.join(model_path, f"segmentation_task", setx)
    output_dir = os.path.join(model_path, "segmentation_task_clean_keep_one_two_wt", setx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted([file for file in os.listdir(input_dir) if "BraTS20" in file])
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 6

    filename = file_list[idx]

    subject = filename.split(".")[0]
    output_path = os.path.join(output_dir, f"{subject}.nii.gz")
    prediction_path = os.path.join(input_dir, f"{subject}.nii.gz")
    segmentation, segmentation_nib = load_volume(prediction_path)
    segmentation_post = segmentation.copy()

    print("Post processing")

    # Keep ONE OR TWO WT
    pred_mask_wt = brats_labels.get_wt(segmentation_post)
    mask_removed_regions_wt = post_process.keep_conn_component_bigger_than_th(pred_mask_wt, th=th)
    elements_to_remove = pred_mask_wt - mask_removed_regions_wt
    segmentation_post[elements_to_remove == 1] = 0

    if setx == "train":
        print("Computing metrics..")
        ground_truth_path = f"/mnt/gpid07/users/laura.mora/datasets/2020/{setx}/no_patch"
        compute_metrics(ground_truth_path, subject, segmentation_post)

    affine_func = segmentation_nib.affine
    save_segmask_as_nifi_volume(segmentation_post, affine_func, output_path)
    print("Result Saved!")