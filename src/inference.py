import os
import sys
import torch
from src.dataset.utils import nifi_volume
import numpy as np

from src.dataset import brats_labels
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset.utils import dataset, visualization
from src.models.io_model import load_model
from src.models.vnet import vnet
from src.test import predict
from src.uncertainty.uncertainty import get_variation_uncertainty, ttd_uncertainty_loop
from src.post_processing import post_process


def load_network(device, model_config, dataset_config):

    n_modalities = dataset_config.getint("n_modalities")
    n_classes = dataset_config.getint("classes")

    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=n_modalities, classes=n_classes,
                        init_features_maps=model_config.getint("init_features_maps"))
    network.to(device)

    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model_path = os.path.dirname(checkpoint_path)

    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    return model, model_path




if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    unc_config = config.get_uncertainty_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 0

    compute_metrics = True
    flag_post_process = True

    model, model_path = load_network(device, model_config, dataset_config)


    data, data_test = dataset.read_brats(dataset_config.get("train_csv"))
    patch_size =  data_test[idx].size

    sampling = dataset_config.get("sampling_method").split(".")[-1]

    ttd = unc_config.getboolean("monte_carlo")
    task = "uncertainty_task" if ttd else "segmentation_task"
    K = unc_config.getint("n_iterations")


    images = data_test[idx].load_mri_volumes(normalize=True)

    # from src.dataset.augmentations.brats_augmentations import zero_mean_unit_variance_normalization
    # images[0,:,:,:] = zero_mean_unit_variance_normalization(images[0,:,:,:])
    # images[1, :, :, :] = zero_mean_unit_variance_normalization(images[1, :, :, :])
    # images[2, :, :, :] = zero_mean_unit_variance_normalization(images[2, :, :, :])
    # images[3, :, :, :] = zero_mean_unit_variance_normalization(images[3, :, :, :])

    if  sampling == "no_patch":

        new_size = (160, 192, 128)
        x_1 = int((patch_size[0] - new_size[0]) / 2)
        x_2 = int(patch_size[0] - (patch_size[0] - new_size[0]) / 2)
        y_1 = int((patch_size[1] - new_size[1]) / 2)
        y_2 = int(patch_size[1] - (patch_size[1] - new_size[1]) / 2)
        z_1 = int((patch_size[2] - new_size[2]) / 2)
        z_2 = int(patch_size[2] - (patch_size[2] - new_size[2]) / 2)
        new_images = images[:, x_1:x_2, y_1:y_2, z_1:z_2]
        images = new_images
        patch_size = new_size

    results = {}

    if ttd:
        prediction_labels_maps, prediction_score_vectors = ttd_uncertainty_loop(model, images, device, K)
        wt_var, tc_var, et_var = get_variation_uncertainty(prediction_score_vectors, patch_size)

        # Get segmentation map by computing the mean of the prediction scores and selecting bigger one
        pred_scores = torch.stack(tuple(prediction_score_vectors)).cpu().numpy()
        pred_scores_mean = np.mean(pred_scores, axis=0)
        prediction_map = np.argmax(pred_scores_mean, axis=1).reshape(patch_size)

        if sampling == "no_patch":
            wt_var = wt_var[:, :, :155]
            tc_var = tc_var[:, :, :155]
            et_var = et_var[:, :, :155]

        results = {"whole": wt_var, "core": tc_var, "enhance": et_var}

    else:
        prediction_four_channels, vector_prediction_scores = predict.predict(model, images, device, monte_carlo=ttd)
        best_scores_map = predict.get_scores_map_from_vector(vector_prediction_scores, patch_size)
        prediction_map = predict.get_prediction_map(prediction_four_channels)


    prediction_map = brats_labels.convert_to_brats_labels(prediction_map)

    if sampling == "no_patch":
        output = np.zeros((240, 240, 155))
        output[x_1:x_2, y_1:y_2, z_1:z_2] = prediction_map
        prediction_map = output
        patch_size = output.shape


    if flag_post_process:
        prediction_map_clean = post_process.opening(prediction_map)
        results["prediction"] = prediction_map_clean
        task = f"{task}_post_processed"
        predict.save_predictions(data_test[idx], results, model_path, task)

    results["prediction"] = prediction_map
    predict.save_predictions(data_test[idx], results, model_path, task)



    if basic_config.getboolean("plot"):
        if ttd:
            visualization.plot_3_view_uncertainty("WT_variance", wt_var, s=round(patch_size[0] / 2), color_map="gray",
                                                  save=True)
            visualization.plot_3_view_uncertainty("TC_variance", tc_var, s=round(patch_size[0] / 2), color_map="gray",
                                                  save=True)
            visualization.plot_3_view_uncertainty("ET_variance", et_var, s=round(patch_size[0] / 2), color_map="gray",
                                                save=True)

        visualization.plot_3_view("final_prediction", prediction_map, s=round(patch_size[0] / 2), discrete=True,
                                  color_map="viridis", save=True)
        visualization.plot_3_view("ground_truth", data_test[idx].load_gt_mask(), s=round(patch_size[0] / 2), discrete=True,
                                  color_map="viridis", save=True)


    if compute_metrics:

        patient_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].seg)
        data_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].flair)

        if os.path.exists(patient_path):
            volume_gt = data_test[idx].load_gt_mask()
            volume = nifi_volume.load_nifi_volume(data_path)
            metrics = compute_wt_tc_et(prediction_map, volume_gt, volume)
            print(metrics)