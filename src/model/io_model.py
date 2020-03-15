import os
import torch

def save_model(model_information, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model_information, path)