import os
from skimage.morphology import remove_small_objects

from src.dataset.utils.nifi_volume import load_nifi_volume_return_nib, save_segmask_as_nifi_volume


def load_segmentation(path):
    return load_nifi_volume_return_nib(path, normalize=False)

if __name__ == "__main__":

    model_path =  "results/checkpoints/model_1596122500/"
    subject  = "BraTS20_Training_001_p0_160x192x128"
    path = os.path.join(model_path, f"segmentation_task/{subject}.nii.gz")

    segmentation, segmentation_nib = load_segmentation(path)

    pred_mask = segmentation > 0
    mask = remove_small_objects(pred_mask, 10000, connectivity=50)
    clean_segmentation = segmentation * mask

    output_dir = os.path.join(model_path, "segmentation_task_clean")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{subject}_clean2.nii.gz")
    save_segmask_as_nifi_volume(clean_segmentation, segmentation_nib.affine, output_path)