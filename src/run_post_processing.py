import csv
import os
from typing import Tuple

from src.compute_metric_results import compute_wt_tc_et
from src.dataset import brats_labels
from src.dataset.utils.nifi_volume import load_nifi_volume_return_nib, save_segmask_as_nifi_volume
from src.dataset.utils.visualization import plot_3_view
from src.post_processing import post_process
from tqdm import tqdm
import numpy as np

def load_volume(path) -> Tuple:
    return load_nifi_volume_return_nib(path, normalize=False)



def compute_metrics(ground_truth_path, subject, segmentation, clean_segmentation):
    gt_path = os.path.join(ground_truth_path, subject, f"{subject}_seg.nii.gz")
    data_path = os.path.join(ground_truth_path, subject, f"{subject}_flair.nii.gz")

    volume_gt, _ = load_volume(gt_path)
    volume, _ = load_volume(data_path)

    metrics_before = compute_wt_tc_et(segmentation, volume_gt, volume)
    metrics_after = compute_wt_tc_et(clean_segmentation, volume_gt, volume)

    print(f"{subject},Before {metrics_before}")
    print(f"{subject},After  {metrics_after}")


if __name__ == "__main__":

    ground_truth_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch"

    model_path =  "results/checkpoints/model_1596122500/"
    input_dir = os.path.join(model_path, "segmentation_task/train_no_post/")
    output_dir = os.path.join(model_path, "segmentation_task_clean_keep_one_connected_each_subregion")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted([file for file in os.listdir(input_dir) if "BraTS20" in file])
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 299

    # for filename in tqdm(file_list, total=len(file_list)):
    filename = file_list[idx]

    subject = filename.split(".")[0]
    output_path = os.path.join(output_dir, f"{subject}.nii.gz")
    prediction_path = os.path.join(input_dir, f"{subject}.nii.gz")
    segmentation, segmentation_nib = load_volume(prediction_path)
    segmentation_post = segmentation.copy()

    print("Post processing")

    pred_mask_wt = brats_labels.get_wt(segmentation_post)
    mask_removed_regions_wt = post_process.keep_bigger_connected_component(pred_mask_wt)
    elements_to_remove = pred_mask_wt - mask_removed_regions_wt
    segmentation_post[elements_to_remove == 1] = 0

    # Keep ONE TC
    pred_mask_tc = brats_labels.get_tc(segmentation_post)
    mask_removed_regions_tc = post_process.keep_bigger_connected_component(pred_mask_tc)
    elements_to_remove = pred_mask_tc - mask_removed_regions_tc
    segmentation_post[elements_to_remove == 1] = 2 # ED

    # ET keep everything or change completly
    segmentation_post = post_process.proportion_tc_et(segmentation_post, th=0.10)

    # pred_mask_et = brats_labels.get_et(segmentation_post)
    # mask_removed_regions_et = post_process.keep_bigger_connected_component(pred_mask_et)
    # elements_to_remove = pred_mask_et - mask_removed_regions_et
    # segmentation_post[elements_to_remove == 1] = 1 # NCR



    print("Computing metrics..")
    compute_metrics(ground_truth_path, subject, segmentation, segmentation_post)

    affine_func = segmentation_nib.affine
    save_segmask_as_nifi_volume(segmentation_post, affine_func, output_path)
    print("Result Saved!")
