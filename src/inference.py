import os
import sys
import torch
from src.dataset.utils import nifi_volume
from tqdm import tqdm
import numpy as np

from src.dataset import brats_labels
from src.compute_metric_results import compute_wt_tc_et
from src.config import BratsConfiguration
from src.dataset.utils import dataset, visualization
from src.models.io_model import load_model
from src.models.vnet import vnet
from src.test import predict
from src.uncertainty.uncertainty import get_variation_uncertainty


if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()
    unc_config = config.get_uncertainty_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    add_padding = True if dataset_config.get("sampling_method").split(".")[-1] == "no_patch" else False

    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=4, classes=4)
    network.to(device)


    checkpoint_path = os.path.join(model_config.get("model_path"), model_config.get("checkpoint"))
    model_path = os.path.dirname(checkpoint_path)

    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)

    _, data_test = dataset.read_brats(dataset_config.get("train_csv"))

    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 0

    patch_size = (240, 240, 240) if add_padding else  data_test[idx].size


    ttd = unc_config.getboolean("monte_carlo")
    K = unc_config.getint("n_iterations")

    compute_metrics = True

    images = data_test[idx].load_mri_volumes(normalize=True)
    results = {}
    if ttd:
        prediction_labels_maps, prediction_score_vectors = [], []

        for _ in tqdm(range(K), desc="Predicting.."):
            prediction_four_channels, vector_prediction_scores = predict.predict(model, images, add_padding,
                                                                                 device, monte_carlo=ttd)
            prediction_labels_maps.append(predict.get_prediction_map(prediction_four_channels))
            prediction_score_vectors.append(vector_prediction_scores)

        wt_var, tc_var, et_var = get_variation_uncertainty(prediction_score_vectors, patch_size)

        # Get segmentation map by computing the mean of the prediction scores and selecting bigger one
        pred_scores = torch.stack(tuple(prediction_score_vectors)).cpu().numpy()
        pred_scores_mean = np.mean(pred_scores, axis=0)
        prediction_map = np.argmax(pred_scores_mean, axis=1).reshape(patch_size)

        if add_padding:
            wt_var = wt_var[:, :, :155]
            tc_var = tc_var[:, :, :155]
            et_var = et_var[:, :, :155]

        results = {"whole": wt_var, "core": tc_var, "enchance": et_var}

    else:
        prediction_four_channels, vector_prediction_scores = predict.predict(model, images, add_padding, device, monte_carlo=ttd)
        best_scores_map = predict.get_scores_map_from_vector(vector_prediction_scores, patch_size)
        prediction_map = predict.get_prediction_map(prediction_four_channels)


    prediction_map = brats_labels.convert_to_brats_labels(prediction_map)
    if add_padding:
        prediction_map = prediction_map[:, :, :155]
    results["prediction"] = prediction_map

    task = "uncertainty_task" if ttd else "segmentation_task"
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