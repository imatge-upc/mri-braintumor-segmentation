import importlib
import sys
import torch
from src.losses.dice_loss import DiceLoss
from src.train.trainer import Trainer, TrainerArgs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

from src.config import BratsConfiguration
from src.dataset import visualization_utils as visualization
from src.dataset.brats_dataset import BratsDataset
from src.dataset.batch_sampler import BratsSampler
from src.dataset import dataset_utils
from src import test
from src.models.vnet import vnet
from src.logging_conf import logger


######## PARAMS
logger.info("Processing Parameters...")

config = BratsConfiguration(sys.argv[1])
model_config = config.get_model_config()
dataset_config = config.get_dataset_config()
basic_config = config.get_basic_config()

patch_size = config.patch_size
tensorboard_logdir = basic_config.get("tensorboard_logs")
batch_size = dataset_config.getint("batch_size")
n_patches = dataset_config.getint("n_patches")
n_patients_per_batch = dataset_config.getint("n_patients_per_batch")
n_classes = dataset_config.getint("classes")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")

######## DATASET
logger.info("Creating Dataset...")

data = dataset_utils.read_brats(dataset_config.get("train_csv"))

# data_train, data_val = train_test_split(data, test_size=0.25, random_state=42)

data_train = data[:n_patches]
data_val = data[:n_patches]

modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                     BratsDataset.t1ce_idx: True}
n_modalities = 4

transforms = T.Compose([T.ToTensor()])

sampling_method = importlib.import_module(dataset_config.get("sampling_method"))

train_dataset = BratsDataset(data_train, modalities_to_use, sampling_method, patch_size, transforms)
train_sampler = BratsSampler(train_dataset, n_patients_per_batch, n_patches)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4)

val_dataset = BratsDataset(data_val, modalities_to_use, sampling_method, patch_size, transforms)
val_sampler = BratsSampler(train_dataset, n_patients_per_batch, n_patches)
val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=4)

if basic_config.getboolean("plot"):
    i, x, y = next(iter(train_loader))
    print(x.shape)
    logger.info('Plotting images')
    visualization.plot_batch_cubes(i, x, y)
    visualization.plot_brain_batch_per_patient(i, train_dataset.data)



######## MODEL
logger.info("Initiating Model...")

if model_config["network"] == "vnet":
    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=n_modalities, classes=n_classes)
    n_params = sum([p.data.nelement() for p in network.parameters()])
    logger.info("Number of params: {}".format(n_params))
else:
    raise ValueError("Bad parameter for network {}".format(model_config.get("network")))


##### TRAIN
logger.info("Start Training")

if basic_config.getboolean("train_flag"):
    network.to(device)

    writer = SummaryWriter(tensorboard_logdir)

    optimizer = torch.optim.SGD(network.parameters(), lr=model_config.getfloat("learning_rate"),
                                momentum=model_config.getfloat("momentum"), weight_decay=model_config.getfloat("weight_decay"))

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=model_config.getfloat("lr_decay"), patience=model_config.getint("patience"))

    criterion = DiceLoss(classes=n_classes)

    args = TrainerArgs(model_config.getint("n_epochs"), device, model_config.get("model_path"))
    trainer = Trainer(args, network, optimizer, criterion, train_loader, val_loader, scheduler, writer)
    trainer.start()

if basic_config.getboolean("test_flag") :
    test.start()

