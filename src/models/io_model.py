import os
import torch

from src.logging_conf import logger


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, is_best, output_path):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        save_path = os.path.join(output_path, f"checkpoint_epoch_{state['epoch']}_val_loss_{state['val_loss']}.pth")
        logger.info(f"Saving a new best to {save_path}")
        torch.save(state, save_path)
    else:
        logger.info("Validation loss did not improve")

def load_model(model, path: str, optimizer=None, resume: bool=False, device: str="cpu"):
    if resume:
        assert optimizer is not None, "Need optimizer to resume training"

    checkpoint = torch.load(path, map_location=torch.device(device))
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']

    model.load_state_dict(checkpoint['state_dict'])

    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, loss
