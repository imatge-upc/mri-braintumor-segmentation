import os
import sys

import numpy as np
import torch
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset import brats_labels
from src.models.vnet import vnet
from src.models.io_model import load_model
from src.test import predict
from src.dataset.utils import dataset, nifi_volume


def inference(window):
    prediction_four_channels, _ = predict.predict(model, window, False, device, monte_carlo=False)
    prediction_map = predict.get_prediction_map(prediction_four_channels)
    return brats_labels.convert_to_brats_labels(prediction_map)


def sliding_window(patient, win_size=64, overlap=0):
    channels, width, height, depth, = patient.shape
    prediction_map = np.zeros((width, height, depth))
    step = win_size - round(win_size * overlap)

    for row in range(0, width-win_size+step, step):
        for column in range(0, height-win_size+step, step):
            for z in range(0, depth-win_size+step, step):

                window = patient[:, row:row+win_size, column:column+win_size, z:z+win_size]
                if window.shape[1:] != (win_size, win_size, win_size):
                    new_array = np.zeros((4, win_size, win_size, win_size))
                    new_array[:, :window.shape[1], :window.shape[2], :window.shape[3]] = window
                    prediction = inference(new_array)
                    prediction_map[row:row+window.shape[1], column:column+window.shape[2], z:z+window.shape[3]] = prediction[:window.shape[1], :window.shape[2], :window.shape[3]]
                else:
                    prediction_map[row:row + win_size, column:column + win_size, z:z + win_size] = inference(window)

    return prediction_map

if __name__ == "__main__":

    print("Loading...")
    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    network = vnet.VNet(elu=True, in_channels=4, classes=4)
    network.to(device)
    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model_path = os.path.dirname(checkpoint_path)

    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    _, data_test = dataset.read_brats(dataset_config.get("train_csv"))
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 0

    images = data_test[idx].load_mri_volumes(normalize=True)
    print("Predicting..")
    prediction_map = sliding_window(images, win_size=64, overlap=0)

    results = {"prediction": prediction_map}
    predict.save_predictions(data_test[idx], results, model_path, "segmentation_task")

    patient_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].seg)
    data_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].flair)

    if os.path.exists(patient_path):
        volume_gt = data_test[idx].load_gt_mask()
        volume = nifi_volume.load_nifi_volume(data_path)
        metrics = compute_wt_tc_et(prediction_map, volume_gt, volume)
        print(metrics)
