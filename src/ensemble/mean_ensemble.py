import os
import sys
import torch
import numpy as np

from src.config import BratsConfiguration
from src.dataset import brats_labels
from src.dataset.utils import dataset
from src.inference import crop_no_patch, return_to_size
from src.models.io_model import load_model
from src.models.unet3d import unet3d
from src.models.vnet import asymm_vnet
from src.test import predict


def _load(network, model_path, checkpoint):
    network.to(device)
    checkpoint_path = os.path.join(model_path, checkpoint)
    model, _, _, _ = load_model(network, checkpoint_path, device, None, False)
    return model


def load_model_1598550861(model_path):
    check = "model_1598550861/checkpoint_epoch_215_val_loss_0.2378825504485875_dice_0.7621174487349105.pth"
    network = asymm_vnet.VNet(non_linearity="prelu", in_channels=4,  classes=4, init_features_maps=32,
                              kernel_size=5, padding=2)
    return _load(network, model_path, check)


def load_model_1598639885(model_path):
    check = "model_1598639885/checkpoint_epoch_198_val_loss_0.19342842820572526_dice_0.8065715717942747.pth"
    network = unet3d.ResidualUNet3D(in_channels=4, out_channels=4, final_sigmoid=False, f_maps=32,
                                    layer_order="crg", num_levels=4, num_groups=4, conv_padding=1)
    return _load(network, model_path, check)


def load_model_1598640035(model_path):
    check = "model_1598640035/checkpoint_epoch_142_val_loss_0.21437616135976087_dice_0.7856238380039416.pth"
    network = unet3d.ResidualUNet3D(in_channels=4, out_channels=4, final_sigmoid=False,
                                    f_maps=32, layer_order="crg", num_levels=4, num_groups=4, conv_padding=1)
    return _load(network, model_path, check)


def load_model_1598640005(model_path):
    check = "model_1598640005/checkpoint_epoch_168_val_loss_0.20105469390137554_dice_0.7989453060986245.pth"
    network = unet3d.UNet3D(in_channels=4, out_channels=4, final_sigmoid=False, f_maps=32,
                            layer_order="crg", num_levels=4, num_groups=4, conv_padding=1)

    return _load(network, model_path, check)


if __name__ == "__main__":

    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampling = dataset_config.get("sampling_method").split(".")[-1]

    models_gen_path = model_config.get("model_path")
    task = "ensemble_segmentation"
    compute_metrics = False

    model_vnet = load_model_1598550861(models_gen_path)
    model_2 = load_model_1598639885(models_gen_path)
    model_3 = load_model_1598640035(models_gen_path)
    model_4 = load_model_1598640005(models_gen_path)

    data, _ = dataset.read_brats(dataset_config.get("val_csv"))

    for idx in range(0, len(data)):
        results = {}

        images = data[idx].load_mri_volumes(normalize=True)

        x_1, x_2, y_1, y_2, z_1, z_2, images, brain_mask, patch_size = crop_no_patch(data[idx].size, images,
                                                                                     data[idx].get_brain_mask(),
                                                                                     sampling)

        _, prediction_four_channels_1 = predict.predict(model_vnet, images, device, False)
        _, prediction_four_channels_2 = predict.predict(model_2, images, device, False)
        _, prediction_four_channels_3 = predict.predict(model_3, images, device, False)
        _, prediction_four_channels_4 = predict.predict(model_4, images, device, False)

        prediction_four_channels_1 = prediction_four_channels_1.view((4, patch_size[0], patch_size[1], patch_size[2]))

        pred_scores = torch.stack([prediction_four_channels_1, prediction_four_channels_2[0], prediction_four_channels_3[0], prediction_four_channels_4[0]]).cpu().numpy()
        pred_scores_mean = np.mean(pred_scores, axis=0)
        prediction_map = np.argmax(pred_scores_mean, axis=0)

        prediction_map = brats_labels.convert_to_brats_labels(prediction_map)
        prediction_map = return_to_size(prediction_map, sampling, x_1, x_2, y_1, y_2, z_1, z_2)

        results["prediction"] = prediction_map
        predict.save_predictions(data[idx], results, models_gen_path, task)
