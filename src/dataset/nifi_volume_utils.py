from scipy.ndimage import zoom
import numpy as np

def resize_volume(data: np.ndarray, resize_percentage:float=0.5) -> np.ndarray:
    # shape = (resize_percentage,) * 3
    shape = (0.267, 0.267, 0.413) # Values to match 256 multiple values for vnet!
    return zoom(data, shape)


def get_one_label_volume(mask: np.ndarray, label: int) -> np.ndarray:
    selector = lambda x: x if x == label else 0
    vfunc = np.vectorize(selector)
    return vfunc(mask)
