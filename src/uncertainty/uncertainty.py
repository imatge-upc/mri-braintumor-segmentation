import os
import sys
import torch
import numpy as np
from src.config import BratsConfiguration
from src.dataset import dataset_utils
from src.dataset.visualization_utils import plot_3_view
from src.models.io_model import load_model
from src.models.vnet import vnet
from src.test.predict import predict


if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    add_padding = True if dataset_config.get("sampling_method").split(".")[-1] == "no_patch" else False


    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=4, classes=4)
    network.to(device)

    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    data_train, data_test = dataset_utils.read_brats(dataset_config.get("train_csv"))

    K = 2
    prediction_maps, prediction_vectors = [], []
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 0
    for _ in range(K):
        prediction_scores, prediction, best_scores_map = predict(model, data_test[idx], add_padding, device, monte_carlo=True, save=False)
        prediction_maps.append(prediction)
        prediction_vectors.append(prediction_scores)

    for i, pred in enumerate(prediction_maps):
        plot_3_view(f"pred_{i}", pred, s=20, save=True)

    # some metric
    y_eval = np.mean(prediction_maps, axis=0).flatten()
    uncertainty_eval = np.var(prediction_maps, axis=0).flatten()
