import sys

import torch
from src.config import BratsConfiguration
from src.dataset.utils import dataset
import numpy as np
from src.dataset.utils.visualization import plot_3_view

if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data, data_test = dataset.read_brats(dataset_config.get("train_csv"))

    images = data_test[0].load_mri_volumes(normalize=True)
    new_size = (160, 192, 128)
    x_1 = int((240-new_size[0])/2)
    x_2 = int(240-(240-new_size[0])/2)
    y_1 = int((240 - new_size[1])/2)
    y_2 = int(240 - (240 - new_size[1])/2)
    new_images = images[:, x_1:x_2, y_1:y_2, :new_size[2]]

    ouput = np.zeros((4, 240, 240, 155))

    ouput[:, x_1:x_2, y_1:y_2,  :new_size[2]] = new_images

    plot_3_view("whole_before", images[0, :, :, :], 100, save=True)
    plot_3_view("cropped", new_images[0, :, :, :], 100, save=True)
    plot_3_view("recovered", ouput[0, :, :, :], 100, save=True)
    print()
