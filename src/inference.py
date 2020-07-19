import os
import sys
import torch
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset.utils import dataset, nifi_volume as nifi_utils
from src.dataset.utils.nifi_volume import load_nifi_volume
from src.dataset.utils.visualization import plot_3_view
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

    _, data_test = dataset.read_brats(dataset_config.get("val_csv"))

    # Use idx to execute predictions in parallel
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    prediction_scores, prediction, best_scores_map = predict(model, data_test[idx], add_padding, device,
                                                                   monte_carlo=False, save=False)

    plot_3_view("prediction", prediction, s=20, save=True)
    plot_3_view("score", best_scores_map, s=20, save=True)

    # compute metrics
    patient_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].seg)
    data_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].flair)



    if os.path.exists(patient_path):
        volume_gt = load_nifi_volume(patient_path, False)
        volume, _ = nifi_utils.load_nifi_volume(data_path)
        metrics = compute_wt_tc_et(prediction, volume_gt, volume)
        print(metrics)