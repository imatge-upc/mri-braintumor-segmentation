import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import visualization_utils as visualization

from dataset.dataset_generator import BratsDataset
from dataset import io
import train, test
from model.vnet import vnet
from logging_conf import logger

######## PARAMS
logger.info('Processing Parameters...')

root_path_server = '/home/usuaris/imatge/laura.mora/dataset_BRATS2019/'
root_path_local = '/Users/lauramora/Documents/MASTER/TFM/Data/'
if os.path.exists(root_path_local):
    root_path = root_path_local
elif os.path.exists(root_path_server):
    root_path = root_path_server
else:
    raise ValueError('No path is working')


path_train = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Training/')
path_val = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Validation/') # TEST (NO GT)

batch_size= 1
train_flag = True
test_flag = False
use_elu = True
network = 'vnet'
use_nll = True
n_epochs = 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

######## DATASET
logger.info('Creating Dataset...')
data, labels = io.get_dataset(path_train)
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.25, random_state=42)

modalities_to_use = { BratsDataset.flair_idx: True,
                      BratsDataset.t1_idx: False,
                      BratsDataset.t2_idx: False,
                      BratsDataset.t1_ce_idx: False}

train_set = BratsDataset(x_train, y_train, modalities_to_use, transforms.Compose([transforms.ToTensor()]), 4)
val_set = BratsDataset(x_val, y_val, modalities_to_use, transforms.Compose([transforms.ToTensor()]), 4)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

######## MODEL
logger.info('Initiating Model...')
if network == 'vnet':
    model = vnet.VNet(elu=use_elu, nll=use_nll)
    n_params = sum([p.data.nelement() for p in model.parameters()])
    logger.info('Number of params: {}'.format(n_params))
else:
    raise ValueError('Bad parameter for network {}'.format(network))

model.to(device)
# model = nn.parallel.DataParallel(model, device_ids=range(torch.cuda.device_count()))

logger.info('Start Training')
if train_flag:
     train.start(model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 epochs=n_epochs,
                 cuda_device=device, model_params={'learning_rate':1e-1, 'momentum':0.99, 'weight_decay':1e-8})

if test_flag:
    test.start()

