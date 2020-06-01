import importlib
import os
import sys

from src.dataset.batch_sampler import BratsSampler
from src.dataset.nifi_volume_utils import save_nifi_volume
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.dataset import dataset_utils
from src.config import BratsConfiguration
from src.dataset.brats_dataset import BratsDataset
from tqdm import tqdm

config = BratsConfiguration(sys.argv[1])
model_config = config.get_model_config()
dataset_config = config.get_dataset_config()
basic_config = config.get_basic_config()


patch_size = config.patch_size
modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                     BratsDataset.t1ce_idx: True}

n_modalities = 4
n_patches = dataset_config.getint("n_patches")
n_patients_per_batch = 1
sampling_method = importlib.import_module(dataset_config.get("sampling_method"))

sampling_name = dataset_config.get("sampling_method").split(".")[-1]
method_path = f"/Users/lauramora/Documents/MASTER/TFM/Data/2019/MICCAI_BraTS_2019_Data_Training_patches/{sampling_name}"


data, labels = dataset_utils.get_dataset(dataset_config.get("path_train"))


train_dataset = BratsDataset(data, labels, modalities_to_use, sampling_method, patch_size, T.Compose([T.ToTensor()]))
train_sampler = BratsSampler(train_dataset, n_patients_per_batch, n_patches)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4)


for patient_id, data_batch, labels_batch in tqdm(train_loader, desc="Computing patches"):
    patient_base_name = train_dataset.get_patient_info(patient_id)["name"]

    for i, (patch_modalities, patch_seg) in enumerate(zip(data_batch, labels_batch)):
        patient_name = f"{patient_base_name}_p{i}_{patch_size[0]}x{patch_size[1]}x{patch_size[2]}"
        patch_path = f"{method_path}/{patient_name}"

        if not os.path.exists(patch_path):
            os.makedirs(patch_path)

        seg_path = f"{patch_path}/{patient_name}_seg.nii.gz"
        flair_path = f"{patch_path}/{patient_name}_flair.nii.gz"
        t1_path = f"{patch_path}/{patient_name}_t1.nii.gz"
        t2_path = f"{patch_path}/{patient_name}_t2.nii.gz"
        t1ce_path = f"{patch_path}/{patient_name}_t1ce.nii.gz"

        save_nifi_volume(patch_seg.numpy(), seg_path)
        save_nifi_volume(patch_modalities[0].numpy(), flair_path)
        save_nifi_volume(patch_modalities[1].numpy(), t1_path)
        save_nifi_volume(patch_modalities[2].numpy(), t2_path)
        save_nifi_volume(patch_modalities[3].numpy(), t1ce_path)
