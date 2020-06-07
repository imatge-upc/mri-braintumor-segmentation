import importlib
import os
import sys

import torch
from src.config import BratsConfiguration
from src.dataset import dataset_utils
from src.dataset.dataset_utils import convert_to_labels
from src.dataset.nifi_volume_utils import save_nifi_volume, load_nifi_volume
from src.dataset.patient import Patient
from src.models.io_model import load_model
from src.models.vnet import vnet
import numpy as np


def _load_data(patient: Patient, sampling_method) -> np.ndarray:
    patient_path = os.path.join(patient.data_path, patient.patch_name)

    flair = load_nifi_volume(os.path.join(patient_path, patient.flair), True)
    t1 = load_nifi_volume(os.path.join(patient_path, patient.t1), True)
    t2 = load_nifi_volume(os.path.join(patient_path, patient.t2), True)
    t1_ce = load_nifi_volume(os.path.join(patient_path, patient.t1ce), True)
    modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))
    # modalities, _ = sampling_method.patching(modalities, None, (160, 192, 128))
    return modalities

def predict(model, patient: Patient, sampling_method, device: torch.device) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        images = torch.from_numpy(_load_data(patient, sampling_method)).unsqueeze(0)
        inputs = images.float().to(device)

        preds = model(inputs)
        output_array = np.asarray(preds[0].max(0)[1].byte().cpu().data)
        output_array = convert_to_labels(output_array)

    output_path = os.path.join(patient.data_path, patient.patch_name, f"{patient.patch_name}_prediction.nii.gz")
    save_nifi_volume(output_array, output_path)
    return output_array


if __name__ == "__main__":
    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    sampling_method = importlib.import_module(dataset_config.get("sampling_method"))

    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=4, classes=4)

    checkpoint_path = "results/checkpoints/checkpoint_epoch_1_val_loss_0.45609837770462036.pth"
    model, _, epoch, loss = load_model(network, checkpoint_path, None, False)

    data = dataset_utils.read_brats(dataset_config.get("train_csv"))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    prediction = predict(model, data[0], sampling_method, device)

    print()