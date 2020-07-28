import numpy as np
import torch

def brats_segmentation_regions() -> dict:
    return {"ET_brats": 4, "ET": 3,  "NCR-NET": 1, "ED": 2}


def get_ncr_net(segmentation_map: np.ndarray) -> np.ndarray:
    regions = brats_segmentation_regions()
    copied_segmentation = _copy_input(segmentation_map)
    copied_segmentation[copied_segmentation != regions["NCR-NET"]] = 0
    return copied_segmentation


def get_ed(segmentation_map: np.ndarray) -> np.ndarray:
    regions = brats_segmentation_regions()
    copied_segmentation = _copy_input(segmentation_map)
    copied_segmentation[copied_segmentation != regions["ED"]] = 0
    return copied_segmentation


def get_et(segmentation_map: np.ndarray) -> np.ndarray:
    """
    ET : enhancing tumors is label 3 in the code, 4 as brats
    :param segmentation_map:
    :return: only label for ET
    """
    regions = brats_segmentation_regions()
    copied_segmentation = _copy_input(segmentation_map)
    unique_values = np.unique(copied_segmentation)
    if max(unique_values) == 3:
        copied_segmentation[copied_segmentation != regions["ET"]] = 0
    else:
        copied_segmentation[copied_segmentation != regions["ET_brats"]] = 0

    return copied_segmentation


def get_wt(segmentation_map: np.ndarray) -> np.ndarray:
    """ WT : entails all regions 4 (ET, NCR, ED ) """
    copied_segmentation = _copy_input(segmentation_map)
    copied_segmentation[copied_segmentation != 0] = 1
    return copied_segmentation


def get_tc(segmentation_map: np.ndarray) -> np.ndarray:
    """ TC : tumor core entails the ET and NCR/NET labels 4 and 1 """
    regions = brats_segmentation_regions()

    copied_segmentation = _copy_input(segmentation_map)
    copied_segmentation[copied_segmentation == regions["ED"]] = 0  # remove edema
    copied_segmentation[copied_segmentation > 0] = 1

    return copied_segmentation

def convert_from_brats_labels(segmentation_map: np.ndarray) -> np.ndarray:
    """Method to convert brats labels as models need consecutive values"""
    regions = brats_segmentation_regions()
    segmentation_map[segmentation_map == regions["ET_brats"]] = regions["ET"]
    return segmentation_map

def convert_to_brats_labels(segmentation_map: np.ndarray) -> np.ndarray:
    """Method to convert recover brats labels encoding"""
    regions = brats_segmentation_regions()
    segmentation_map[segmentation_map == regions["ET"]] = regions["ET_brats"]
    return segmentation_map


def _copy_input(input):
    if torch.is_tensor(input):
        return input.detach().clone()
    else:
        return input.copy()
