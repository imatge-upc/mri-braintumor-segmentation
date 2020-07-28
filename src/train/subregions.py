import os
import sys
import torch
from src.dataset.utils import nifi_volume
from tqdm import tqdm
import numpy as np

from src.dataset import brats_labels
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset.utils import dataset, visualization
from src.models.io_model import load_model
from src.models.vnet import vnet
from src.test import predict
from src.uncertainty.uncertainty import get_variation_uncertainty


if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    unc_config = config.get_uncertainty_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    add_padding = True if dataset_config.get("sampling_method").split(".")[-1] == "no_patch" else False

    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=4, classes=4)
    network.to(device)


    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model_path = os.path.dirname(checkpoint_path)

    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    _, data_test = dataset.read_brats(dataset_config.get("train_csv"))

    patient = data_test[0]
    patch_size = patient.size

    images = patient.load_mri_volumes(normalize=True)

    prediction_four_channels, vector_prediction_scores = predict.predict(model, images, False, device, monte_carlo=False)
    pred = prediction_four_channels.max(1)[1]

    et = brats_labels.get_et(pred)
    tc = brats_labels.get_tc(pred)
    wt = brats_labels.get_wt(pred)


    print()