import numpy as np


def get_one_label_volume(mask: np.ndarray, label: int) -> np.ndarray:
    selector = lambda x: x if x == label else 0
    vfunc = np.vectorize(selector)
    return vfunc(mask)
