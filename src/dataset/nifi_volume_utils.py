from scipy.ndimage import zoom
import numpy as np

def resize_volume(data: np.ndarray, resize_percentage:float=0.5) -> np.ndarray:
    # shape = (resize_percentage,) * 3
    shape = (0.267, 0.267, 0.413) # Values to match 256 multiple values for vnet!
    return zoom(data, shape)