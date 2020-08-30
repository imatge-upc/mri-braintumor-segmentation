import os

from src.dataset.utils import dataset
from src.dataset.utils.nifi_volume import load_nifi_volume_return_nib, save_segmask_as_nifi_volume
from src.uncertainty.filter_by_threshold import filter_by_threshold_eval_regions
from tqdm import tqdm


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_niigz(volume_name, path_to_volume):
    path = os.path.join(path_to_volume, volume_name)
    return load_nifi_volume_return_nib(path, normalize=False)


if __name__ == "__main__":
    setx = "train"
    model_id = "model_1597063224/101_epch"

    dataset_csv_path = f"/Users/lauramora/Documents/MASTER/TFM/Data/2020/{setx}/no_patch/brats20_data.csv"
    model_path = f"/Users/lauramora/Documents/MASTER/TFM/Code/BrainTumorSegmentation/results/checkpoints/{model_id}/"

    task = "uncertainty_task_normalized"
    input_dir = os.path.join(model_path, task)
    output_dir = os.path.join(model_path, f"{task}/filtered/")
    create_dir(output_dir)


    data, _ = dataset.read_brats(dataset_csv_path, lgg_only=False)
    patient_names = [p.patient for p in data]

    thresholds = [75, 50, 25, 10, 0]

    for patient_name in tqdm(patient_names, total=len(patient_names), desc="Filtering_by_T"):
        # Read patient info
        wt_unc, wt_nib = read_niigz(f"{patient_name}_unc_whole.nii.gz", input_dir)
        et_unc, et_nib = read_niigz(f"{patient_name}_unc_enhance.nii.gz", input_dir)
        tc_unc, tc_nib = read_niigz(f"{patient_name}_unc_core.nii.gz", input_dir)
        seg_map, seg_nib = read_niigz(f"{patient_name}.nii.gz", input_dir)

        for T in thresholds:
            output_path = os.path.join(output_dir, str(T))
            create_dir(output_path)
            output_path_with_name = os.path.join(output_path, f"{patient_name}.nii.gz")

            filtered_map = filter_by_threshold_eval_regions(T, seg_map, wt_unc, tc_unc, et_unc)

            save_segmask_as_nifi_volume(filtered_map, seg_nib.affine, output_path_with_name)


