import numpy as np
import os
from tqdm import tqdm

from src.dataset.utils import dataset
from src.dataset.utils import nifi_volume
from src.dataset.utils.nifi_volume import save_segmask_as_nifi_volume
from src.logging_conf import logger
from src.compute_metric_results import compute_wt_tc_et


def majority_voting(segmentation_predictions: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    rows, columns, depth = segmentation_predictions.shape[1], segmentation_predictions.shape[2], \
                           segmentation_predictions.shape[3]

    majority_voting = np.zeros((rows, columns, depth))

    for x in range(0, rows):
        for y in range(0, columns):
            for z in range(0, depth):
                if brain_mask[x, y, z] == 1:
                    values, counts = np.unique(segmentation_predictions[:, x, y, z], return_counts=True)
                    idx = counts.argmax()
                    majority_voting[x, y, z] = values[idx]
                else:
                    majority_voting[x, y, z] = 0

    return majority_voting


def read_preds_from_models(model_list: list, patient_name: str) -> np.ndarray:
    seg_maps = [nifi_volume.load_nifi_volume(os.path.join(model_path, patient_name), normalize=False)
                for model_path in model_list]

    return np.stack(seg_maps)


if __name__ == "__main__":
    gen_path = "/mnt/gpid07/users/laura.mora"

    compute_metrics = False
    setx = "train"
    csv = "brats20_val.csv" if setx == "validation" else "brats20_data.csv"

    task = f"segmentation_task/{setx}"
    dataset_csv_path = f"{gen_path}/datasets/{setx}/no_patch/{csv}"

    models = ["model_1598550861", "model_1598639885", "model_1598640035", "model_1598640005"]
    check_path = f"{gen_path}results/checkpoints/"
    models = list(map(lambda x:  os.path.join(check_path, x, task), models))

    output_dir = os.path.join(check_path, f"{task}/ensemble_predictions/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data, _ = dataset.read_brats(dataset_csv_path, lgg_only=False)

    for patient in tqdm(data, total=len(data), desc="Ensemble prediction"):
        patient_name = patient.patient

        seg_maps = read_preds_from_models(models, f"{patient_name}.nii.gz")
        ensemble_map = majority_voting(seg_maps, patient.get_brain_mask())

        output_path_with_name = os.path.join(output_dir, f"{patient_name}.nii.gz")
        save_segmask_as_nifi_volume(ensemble_map, patient.get_affine(), output_path_with_name)

        if compute_metrics:
            patient_path = os.path.join(patient.data_path, patient.patch_name, patient.seg)
            data_path = os.path.join(patient.data_path, patient.patch_name, patient.flair)

            if os.path.exists(patient_path):
                volume_gt = patient.load_gt_mask()
                volume = nifi_volume.load_nifi_volume(data_path)
                metrics = compute_wt_tc_et(ensemble_map, volume_gt, volume)
                logger.info(f"{patient.patient} | {metrics}")
