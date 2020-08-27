from tqdm import tqdm

from src.test import predict



def ttd_uncertainty_loop(model, images, device, K=2):

    prediction_labels_maps, prediction_score_vectors = [], []

    for _ in tqdm(range(K), desc="Predicting.."):
        prediction_four_channels, vector_prediction_scores = predict.predict(model, images, device, monte_carlo=True)

        prediction_labels_maps.append(predict.get_prediction_map(prediction_four_channels))
        prediction_score_vectors.append(vector_prediction_scores)

    return prediction_labels_maps, prediction_score_vectors