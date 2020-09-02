from tqdm import tqdm
from src.dataset.utils.visualization import plot_3_view
from src.test import predict
from src.dataset.augmentations import color_augmentations, spatial_augmentations
import numpy as np


def _get_transforms():
    return [color_augmentations.RandomIntensityShift(), color_augmentations.RandomIntensityScale(),
            color_augmentations.RandomGaussianNoise(p=1, noise_variance=(0, 0.5))]


def tta_uncertainty_loop(model, images, device, brain_mask, iterations=2):
    prediction_labels_maps, prediction_score_vectors = [], []

    data_transformations = _get_transforms()
    range_transforms = range(0, len(data_transformations))

    for i in tqdm(range(iterations), desc="Predicting.."):
        random_transform_idx = np.random.choice(range_transforms)
        transform = data_transformations[random_transform_idx]

        subject, _, _ = transform((images, None, brain_mask))

        prediction_four_channels, vector_prediction_scores = predict.predict(model, subject.astype(float), device,
                                                                             monte_carlo=False)
        pred_map = predict.get_prediction_map(prediction_four_channels)

        prediction_labels_maps.append(pred_map)
        prediction_score_vectors.append(vector_prediction_scores)

    return prediction_labels_maps, prediction_score_vectors
