import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def start(model, dataloader, device):
    model.eval()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    fig.suptitle('predicted_mask - original_mask')
    for i, images, labels_batch in tqdm(dataloader):
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()


        ax1.imshow(np.squeeze(batch_preds), cmap='gray')
        ax2.imshow(np.squeeze(labels_batch), cmap='gray')
        break