import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


def opening(segmentation_mask: np.ndarray, kernel_size: tuple=(8,8,8)):

    kernel =  np.ones(kernel_size)
    mask =  ndimage.binary_opening(segmentation_mask, structure=kernel).astype(int)
    return segmentation_mask * mask

def remove_small_elements(segmentation_mask, min_size=1000):

    pred_mask = segmentation_mask > 0

    mask = remove_small_objects(pred_mask, min_size=min_size)

    clean_segmentation = segmentation_mask * mask

    return clean_segmentation


def keep_bigger_connected_component(segmentation_mask):
    labels = label(segmentation_mask)

    maxArea = 0
    for region in regionprops(labels):
        if region.area > maxArea:
            maxArea = region.area

    mask = remove_small_objects(labels, maxArea - 1)

    return np.asarray(mask > 0, np.uint8)


def keep_conn_component_bigger_than_th(segmentation_mask, th=8):
    labels = label(segmentation_mask)

    areas = sorted([region.area for region in regionprops(labels)], reverse=True) # big to small
    if len(areas) > 1:
        diff_big = (areas[1] / areas[0]) * 100
        area = areas[1] if diff_big > th else areas[0]
    else:
        area = areas[0]

    mask = remove_small_objects(labels, area - 1)

    return np.asarray(mask > 0, np.uint8)


if __name__ == "__main__":

    import nibabel as nib
    import os
    # Read volume
    gt_path = "../../Data/2020/train/no_patch/"
    result_path = "../BrainTumorSegmentation/results/checkpoints/model_1596122500/segmentation_task/train_no_post"
    patient = "BraTS20_Training_070"

    flair = nib.load(os.path.join(gt_path, patient, f"{patient}_flair.nii.gz"))
    seg = nib.load(os.path.join(gt_path, patient, f"{patient}_seg.nii.gz"))
    pred = nib.load(os.path.join(result_path, f"{patient}.nii.gz"))

    arr_flair = flair.get_fdata()
    arr_seg = seg.get_fdata()
    arr_pred = pred.get_fdata()

    # Use WT
    pred_mask = np.asarray(arr_pred > 0, np.uint8)
    mask_removed_regions = keep_bigger_connected_component(pred_mask)
    final_segmentation_one_big = arr_pred * mask_removed_regions

    mask_removed_regions = keep_conn_component_bigger_than_th(pred_mask)
    final_segmentation_two_big = arr_pred * mask_removed_regions
