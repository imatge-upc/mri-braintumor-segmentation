import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects


def opening(segmentation_mask: np.ndarray, kernel_size: tuple=(8,8,8)):

    kernel =  np.ones(kernel_size)
    mask =  ndimage.binary_opening(segmentation_mask, structure=kernel).astype(int)
    return segmentation_mask * mask


def remove_small_elements(segmentation_mask, min_size=1000):

    pred_mask = segmentation_mask > 0

    mask = remove_small_objects(pred_mask, min_size=min_size)

    clean_segmentation = segmentation_mask * mask

    return clean_segmentation
