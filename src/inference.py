import os
import sys
import torch
from src.dataset.utils import nifi_volume
import numpy as np

from src.dataset import brats_labels
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset.utils import dataset
from src.models.io_model import load_model
from src.models.unet3d import unet3d
from src.models.vnet import vnet
from src.test import predict
from src.uncertainty.uncertainty import get_variation_uncertainty, ttd_uncertainty_loop, get_entropy_uncertainty
from src.post_processing import post_process
from src.logging_conf import logger


def load_network(device, model_config, dataset_config, which_net):

    n_modalities = dataset_config.getint("n_modalities")
    n_classes = dataset_config.getint("classes")

    if which_net == "vnet":
        network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=n_modalities, classes=n_classes,
                            init_features_maps=model_config.getint("init_features_maps"))
    elif which_net == "3dunet_residual":
        network = unet3d.ResidualUNet3D(in_channels=n_modalities, out_channels=n_classes, final_sigmoid=False, # so i get a softmax
                                    f_maps=model_config.getint("init_features_maps"), layer_order="crg",
                                    num_levels=4, num_groups=4,conv_padding=1)

    elif which_net == "3dunet":
         network = unet3d.UNet3D(in_channels=n_modalities, out_channels=n_classes, final_sigmoid=False,
                                    f_maps=model_config.getint("init_features_maps"), layer_order="crg",
                                    num_levels=4, num_groups=4,conv_padding=1)

    else:
        raise ValueError(f"bad network {which_net}")
    network.to(device)

    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model_path = os.path.dirname(checkpoint_path)

    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    return model, model_path


def crop_no_patch(patch_size, images, sampling):

    if sampling == "no_patch":
        new_size = (160, 192, 128)
        x_1 = int((patch_size[0] - new_size[0]) / 2)
        x_2 = int(patch_size[0] - (patch_size[0] - new_size[0]) / 2)
        y_1 = int((patch_size[1] - new_size[1]) / 2)
        y_2 = int(patch_size[1] - (patch_size[1] - new_size[1]) / 2)
        z_1 = int((patch_size[2] - new_size[2]) / 2)
        z_2 = int(patch_size[2] - (patch_size[2] - new_size[2]) / 2)
        new_images = images[:, x_1:x_2, y_1:y_2, z_1:z_2]
        return x_1, x_2, y_1, y_2, z_1, z_2, new_images, new_size

    else:
        x_1, x_2 = 0, patch_size[0]
        y_1, y_2 = 0, patch_size[1]
        z_1, z_2 = 0, patch_size[2]
        return x_1, x_2, y_1, y_2, z_1, z_2, images, patch_size

def return_to_size(volume, sampling, x_1, x_2, y_1, y_2, z_1, z_2, final_size=(240, 240,155)):

    if sampling == "no_patch":
        output = np.zeros(final_size)
        output[x_1:x_2, y_1:y_2, z_1:z_2] = volume
        return output

    else:
        return volume


if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    unc_config = config.get_uncertainty_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    compute_metrics = False
    flag_post_process = False

    model, model_path = load_network(device, model_config, dataset_config, model_config["network"])

    data, data_test = dataset.read_brats(dataset_config.get("train_csv"))
    data_test.extend(data)


    sampling = dataset_config.get("sampling_method").split(".")[-1]

    ttd = unc_config.getboolean("monte_carlo")
    task = "uncertainty_task" if ttd else "segmentation_task"
    K = unc_config.getint("n_iterations")

    for idx in range(0, len(data_test)):

        patch_size = data_test[idx].size

        images = data_test[idx].load_mri_volumes(normalize=True)

        x_1, x_2, y_1, y_2, z_1, z_2, images, patch_size = crop_no_patch(patch_size, images, sampling)

        results = {}

        if ttd:
            prediction_labels_maps, prediction_score_vectors = ttd_uncertainty_loop(model, images, device, K)
            # Get segmentation map by computing the mean of the prediction scores and selecting bigger one
            pred_scores = torch.stack(tuple(prediction_score_vectors)).cpu().numpy()
            pred_scores_mean = np.mean(pred_scores, axis=0)
            prediction_map = np.argmax(pred_scores_mean, axis=1).reshape(patch_size)

            wt_var, tc_var, et_var = get_variation_uncertainty(prediction_score_vectors, patch_size)
            global_unc = get_entropy_uncertainty(prediction_score_vectors, patch_size)


            wt_var = return_to_size(wt_var, sampling, x_1, x_2, y_1, y_2, z_1, z_2)
            tc_var = return_to_size(tc_var, sampling, x_1, x_2, y_1, y_2, z_1, z_2)
            et_var = return_to_size(et_var, sampling, x_1, x_2, y_1, y_2, z_1, z_2)
            global_unc = return_to_size(global_unc, sampling, x_1, x_2, y_1, y_2, z_1, z_2)

            results = {"whole": wt_var, "core": tc_var, "enhance": et_var, "entropy": global_unc}

        else:
            prediction_four_channels, vector_prediction_scores = predict.predict(model, images, device, monte_carlo=ttd)
            if model_config["network"] == "vnet":
                best_scores_map = predict.get_scores_map_from_vector(vector_prediction_scores, patch_size)
            else:
                best_scores_map = vector_prediction_scores

            prediction_map = predict.get_prediction_map(prediction_four_channels)

        prediction_map = brats_labels.convert_to_brats_labels(prediction_map)
        prediction_map = return_to_size(prediction_map, sampling, x_1, x_2, y_1, y_2, z_1, z_2)

        if flag_post_process:
            segmentation_post = prediction_map.copy()
            pred_mask_wt = brats_labels.get_wt(segmentation_post)
            mask_removed_regions_wt = post_process.keep_conn_component_bigger_than_th(pred_mask_wt, th=4)
            elements_to_remove = pred_mask_wt - mask_removed_regions_wt
            segmentation_post[elements_to_remove == 1] = 0
            results["prediction"] = segmentation_post
            task = f"{task}_post_processed"
            predict.save_predictions(data_test[idx], results, model_path, task)

        results["prediction"] = prediction_map
        predict.save_predictions(data_test[idx], results, model_path, task)

        if compute_metrics:

            patient_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].seg)
            data_path = os.path.join(data_test[idx].data_path, data_test[idx].patch_name, data_test[idx].flair)

            if os.path.exists(patient_path):
                volume_gt = data_test[idx].load_gt_mask()
                volume = nifi_volume.load_nifi_volume(data_path)
                metrics = compute_wt_tc_et(prediction_map, volume_gt, volume)
                logger.info(f"{data_test[idx].patient} | {metrics}")
