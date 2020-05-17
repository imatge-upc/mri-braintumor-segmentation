import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time

from config import get_configuration, get_dataset_path
from dataset.dataset_generator import BratsDataset
from dataset import io
import train, test
from models.vnet import vnet
from logging_conf import logger

######## PARAMS
logger.info('Processing Parameters...')

config = get_configuration(sys.argv[1])
path_train, path_test = get_dataset_path(config.get("dataset", "dataset_root_path_local"),
                                         config.get("dataset", "dataset_root_path_server"))

logs = f"{config.get('basics', 'tensorboard_logs')}_{round(time.time())}"
logger.debug(logs)
if not os.path.exists(logs):
    os.makedirs(logs)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Device: {device}")

######## DATASET
logger.info('Creating Dataset...')
data, labels = io.get_dataset(path_train)
# results = results[:10]
# labels = labels[:10]

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.25, random_state=42)

modalities_to_use = { BratsDataset.flair_idx: True,
                      BratsDataset.t1_idx: False,
                      BratsDataset.t2_idx: False,
                      BratsDataset.t1_ce_idx: False}

train_set = BratsDataset(x_train, y_train, modalities_to_use, transforms.Compose([transforms.ToTensor()]), label=None)
val_set = BratsDataset(x_val, y_val, modalities_to_use, transforms.Compose([transforms.ToTensor()]), label=None)

train_loader = DataLoader(train_set, batch_size=config.getint('model', 'batch_size'), shuffle=False)
val_loader = DataLoader(val_set, batch_size=config.getint('model', 'batch_size'), shuffle=False)

######## MODEL
logger.info('Initiating Model...')
if config.get('model', 'network') == 'vnet':
    model = vnet.VNet(elu=config.getboolean('model', 'use_elu'),
                      batch_size=config.getint('model', 'batch_size'),
                      labels=3)

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logger.info('Number of params: {}'.format(n_params))
else:
    raise ValueError('Bad parameter for network {}'.format(config.getint('model', 'network')))


logger.info('Start Training')
if config.getboolean('basics', 'train_flag'):
    model.to(device)
    writer = SummaryWriter(logs)

    train_params = {'learning_rate': config.getfloat('model', 'learning_rate'),
                    'momentum': config.getfloat('model', 'momentum'),
                    'weight_decay': config.getfloat('model', 'weight_decay'),
                    'scheduler': config.getboolean('model', 'scheduler'),
                    'output_path': config.get('model', 'model_path')
                    }

    train.start(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.getint('model', 'n_epochs'),
                cuda_device=device,
                writer=writer,
                model_params=train_params)

if config.getboolean('basics', 'test_flag') :
    test.start()

