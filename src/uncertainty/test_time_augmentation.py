from tqdm import tqdm
from src.test import predict
from src.dataset.augmentations import color_augmentations
import numpy as np
from src.dataset.utils.visualization import plot_3_view


def _get_transforms():
    return [color_augmentations.RandomIntensityShift(min=-0.1, max=0.1),  # 0
            color_augmentations.RandomIntensityScale(min=0.9, max=1.1),  # 1
            color_augmentations.RandomGaussianNoise(p=1, noise_variance=(0, 0.5))]  # 3


def tta_uncertainty_loop(model, images, device, brain_mask, iterations=2, monte_carlo=False):
    prediction_labels_maps, prediction_score_vectors = [], []

    data_transformations = _get_transforms()
    range_transforms = range(0, len(data_transformations))

    for i in tqdm(range(iterations), desc="Predicting.."):
        random_transform_idx = np.random.choice(range_transforms)
        transform = data_transformations[random_transform_idx]

        subject, _, _ = transform((images, None, brain_mask))

        prediction_four_channels, vector_prediction_scores = predict.predict(model, subject.astype(float), device,
                                                                             monte_carlo=monte_carlo)
        pred_map = predict.get_prediction_map(prediction_four_channels)

        plot_3_view(f"pred_map_{i}_{random_transform_idx}", pred_map[:, :, :], 40, save=True)
        plot_3_view(f"subject_{i}_{random_transform_idx}", subject[0, :, :, :], 40, save=True)

        prediction_labels_maps.append(pred_map)
        prediction_score_vectors.append(vector_prediction_scores)

    return prediction_labels_maps, prediction_score_vectors
