import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import os

from losses.losses import SoftDiceLoss
from metrics.training_metrics import AverageMeter, dice_coefficient
from dataset import visualization_utils as visualization
from logging_conf import logger
from model.io_model import save_checkpoint

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc):
    logger.info(f'epoch: {epoch} | train_loss: {train_loss:.2f} | train_acc: {train_acc} | val_loss {val_loss:.2f} | val_acc {val_acc:.2f}')


def _step(model, criterion, inputs, targets, device):

    inputs = inputs.float().to(device)
    targets = targets.float().to(device)

    outputs = model(inputs)

    loss = criterion(outputs, targets)
    acc = dice_coefficient(outputs.cpu(), targets.cpu())

    return acc, loss


def training_step(model, train_loader, optimizer, criterion, device, epoch, writer, plot=False):
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    i = 0
    for data_batch, labels_batch, paths in tqdm(train_loader, desc="Training epoch"):
        if plot:
            logger.info('Plotting images')
            visualization.plot_batch(data_batch, labels_batch)

        acc, loss = _step(model, criterion, data_batch, labels_batch, device)

        losses.update(loss.cpu(), data_batch.size(0))
        accuracies.update(acc, data_batch.size(0))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Training Accuracy', acc.item(), epoch * len(train_loader) + i)
        i +=1

    return losses.avg, accuracies.avg


@torch.no_grad()
def validation_step(model, val_loader, criterion, device, epoch, writer):

    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    i = 0
    for data_batch, labels_batch, paths in tqdm(val_loader, desc="Validation epoch"):
        acc, loss = _step(model, criterion, data_batch, labels_batch, device)
        losses.update(loss.cpu(), data_batch.size(0))
        accuracies.update(acc, data_batch.size(0))

        writer.add_scalar('Validation loss', loss.item(), epoch * len(val_loader) + i)
        writer.add_scalar('Validation accuracy', acc.item(), epoch * len(val_loader) + i)
        i += 1

    return losses.avg, accuracies.avg


def start(model, train_loader, val_loader, epochs, cuda_device, writer, model_params=None):

    if not model_params:
        raise ValueError('Need model parameters!')

    best_accuracy = 0
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=model_params['learning_rate'],
                                momentum=model_params['momentum'],
                                weight_decay=model_params['weight_decay'])

    if model_params["scheduler"]:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=model_params["lr_decay"],
                                                   patience=model_params["patience"])

    loss_function = SoftDiceLoss()

    for epoch in range(start_epoch, epochs):
        logger.info(f'Epoch {epoch}')
        train_loss, train_acc = training_step(model, train_loader, optimizer, loss_function, cuda_device, epoch, writer)
        val_loss, val_acc = validation_step(model, val_loader, loss_function, cuda_device, epoch, writer)

        if model_params["scheduler"]:
            scheduler.step(val_loss)

        _epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc)
        # Save checkpoint when val_acc increases
        is_best = bool(val_acc > best_accuracy)
        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy,
            'loss': val_loss
        }, is_best, model_params["output_path"])

