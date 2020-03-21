import torch
from tqdm import tqdm
from dataset import visualization_utils as visualization
from losses import losses
from logging_conf import logger


def training_step(model, train_loader, optimizer, device, plot=False):
    model.train()
    for data_batch, labels_batch, paths in tqdm(train_loader):
        if plot:
            logger.info('Plotting images')
            visualization.plot_batch(data_batch, labels_batch)

        data_batch = data_batch.requires_grad_()
        data_batch = data_batch.float().to(device)
        logger.debug('Model called')
        outputs = model(data_batch)

        target = labels_batch.view(labels_batch.numel())
        target = target.long().to(device)
        target = target.view(target.numel())

        logger.debug('Loss called')
        loss = losses.nll_loss(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation_step(model, val_loader, cuda_device):
    model.eval()
    with torch.no_grad():
        for i_val, (data_batch, labels_batch, paths) in tqdm(enumerate(val_loader)):
            images_val = data_batch.to(cuda_device)
            labels_val = labels_batch.to(cuda_device)


def start(model, train_loader, val_loader, epochs, cuda_device, model_params=None):
    if not model_params:
        raise ValueError('Need model parameters!')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=model_params['learning_rate'],
                                momentum=model_params['momentum'],
                                weight_decay=model_params['weight_decay'])


    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}')
        training_step(model, train_loader, optimizer, cuda_device, plot=False)
       #  validation_step(model, val_loader, cuda_device)

