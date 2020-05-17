import os
import torch

from logging_conf import logger


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, is_best, output_path):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        save_path = os.path.join(output_path,
                                 f"checkpoint_epoch_{state['epoch']}_acc_{state['best_accuracy']}_loss_{state['loss']}.pth")
        logger.info(f"Saving a new best to {save_path}")
        torch.save(state, save_path)
    else:
        logger.info("Validation Accuracy did not improve")