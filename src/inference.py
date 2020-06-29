import os
import sys

import torch
from src.config import BratsConfiguration
from src.dataset import dataset_utils
from src.dataset.dataset_utils import convert_to_labels
from src.dataset.nifi_volume_utils import save_segmask_as_nifi_volume, load_nifi_volume
from src.dataset.patient import Patient
from src.models.io_model import load_model
from src.models.vnet import vnet
import numpy as np


def _load_data(patient: Patient) -> np.ndarray:
    patient_path = os.path.join(patient.data_path, patient.patch_name)

    flair = load_nifi_volume(os.path.join(patient_path, patient.flair), True)
    t1 = load_nifi_volume(os.path.join(patient_path, patient.t1), True)
    t2 = load_nifi_volume(os.path.join(patient_path, patient.t2), True)
    t1_ce = load_nifi_volume(os.path.join(patient_path, patient.t1ce), True)
    modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

    return modalities

def predict(model, patient: Patient, add_padding: bool, device: torch.device) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        images = _load_data(patient)
        if add_padding:
            new_array = np.zeros((4, 240, 240, 240))
            new_array[:,  :images.shape[1], :images.shape[2], :images.shape[3]] = images
            images = new_array

        images = torch.from_numpy(images).unsqueeze(0)
        inputs = images.float().to(device)

        preds = model(inputs)
        output_array = np.asarray(preds[0].max(0)[1].byte().cpu().data)
        output_array = convert_to_labels(output_array)

    output_path = os.path.join(patient.data_path, patient.patch_name, f"{patient.patch_name}_prediction.nii.gz")
    flair_path = os.path.join(patient.data_path, patient.patch_name, patient.flair)
    print(f"Saving prediction to: {output_path}")
    if add_padding:
        output_array = output_array[:, :, :155]
    save_segmask_as_nifi_volume(output_array, flair_path, output_path)

    return output_array


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
    model, _, epoch, loss = load_model(network, checkpoint_path, device, None, False)

    _, data_test = dataset_utils.read_brats(dataset_config.get("train_csv"))

    # Use idx to execute predictions in parallel
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    prediction = predict(model, data_test[idx], add_padding, device)

    # ADD metrics