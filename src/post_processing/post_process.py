import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from src.dataset import brats_labels


def opening(segmentation_mask: np.ndarray, kernel_size: tuple=(8,8,8)) ->  np.ndarray:

    kernel =  np.ones(kernel_size)
    mask =  ndimage.binary_opening(segmentation_mask, structure=kernel).astype(int)
    return segmentation_mask * mask

def remove_small_elements(segmentation_mask: np.ndarray, min_size: int=1000) ->  np.ndarray:

    pred_mask = segmentation_mask > 0

    mask = remove_small_objects(pred_mask, min_size=min_size)

    clean_segmentation = segmentation_mask * mask

    return clean_segmentation


def keep_bigger_connected_component(segmentation_mask:  np.ndarray) ->  np.ndarray:
    labels = label(segmentation_mask)

    maxArea = 0
    for region in regionprops(labels):
        if region.area > maxArea:
            maxArea = region.area

    mask = remove_small_objects(labels, maxArea - 1)

    return np.asarray(mask > 0, np.uint8)



def keep_conn_component_bigger_than_th(segmentation_mask: np.ndarray, th: int=8) -> np.ndarray:
    labels = label(segmentation_mask)

    areas = sorted([region.area for region in regionprops(labels)], reverse=True) # big to small
    if len(areas) > 1:
        diff_big = (areas[1] / areas[0]) * 100
        area = areas[1] if diff_big > th else areas[0]
    else:
        area = areas[0]

    mask = remove_small_objects(labels, area - 1)

    return np.asarray(mask > 0, np.uint8)


def proportion_tc_et(prediction: np.ndarray, th: float=0.10) -> np.ndarray:

    """
    Mean prop for tc et in LGG in 0.08 with std 0.13 --> test th=10
    :param tc_mask:
    :param et_mask:
    :param th:
    :return:
    """
    et_mask = brats_labels.get_et(prediction)
    tc_mask = brats_labels.get_tc(prediction)

    et_tc_prop = np.count_nonzero(et_mask) / np.count_nonzero(tc_mask)

    if et_tc_prop <= th:
        prediction[prediction == 4] = 1 # convert to NCR/NET

    return prediction


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
