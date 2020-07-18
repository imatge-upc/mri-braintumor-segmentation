import importlib
import sys
import torch
from src.dataset.train_val_split import train_val_split

from src.losses.dice_loss import DiceLoss
from src.models.io_model import load_model
from src.train.trainer import Trainer, TrainerArgs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from src.config import BratsConfiguration
from src.dataset.loaders.brats_dataset_whole_volume import BratsDataset
from src.dataset.loaders.brats_dataset_patches import BratsDatasetPatches


from src.dataset.utils import dataset, visualization as visualization
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
checkpoint_path = model_config.get("checkpoint")

batch_size = dataset_config.getint("batch_size")
n_patches = dataset_config.getint("n_patches")
n_classes = dataset_config.getint("classes")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")


######## DATASET
logger.info("Creating Dataset...")

data, data_test = dataset.read_brats(dataset_config.get("train_csv"))
# data_train, data_val = train_val_split(data, val_size=0.1)

data_train = [data[123]]  * n_patches
data_val = [data[123]] * n_patches

n_modalities = dataset_config.getint("n_modalities") # like color channels
modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                     BratsDataset.t1ce_idx: True}

transforms = T.Compose([T.ToTensor()])

sampling_method = importlib.import_module(dataset_config.get("sampling_method"))

train_dataset = BratsDatasetPatches(data_train, modalities_to_use, sampling_method, patch_size, transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = BratsDatasetPatches(data_val, modalities_to_use, sampling_method, patch_size, transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if basic_config.getboolean("plot"):
    i, x, y = next(iter(train_loader))
    print(x.shape)
    logger.info('Plotting images')
    # visualization.plot_batch_cubes(i, x, y)
    visualization.plot_brain_batch_per_patient(i, train_dataset.data)



######## MODEL
logger.info("Initiating Model...")

if model_config["network"] == "vnet":
    network = vnet.VNet(elu=model_config.getboolean("use_elu"), in_channels=n_modalities, classes=n_classes)
    n_params = sum([p.data.nelement() for p in network.parameters()])
    logger.info("Number of params: {}".format(n_params))
else:
    raise ValueError("Bad parameter for network {}".format(model_config.get("network")))


if basic_config.getboolean("train_flag"):
    ##### TRAIN
    logger.info("Start Training")
    network.to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr=model_config.getfloat("learning_rate"),
                                momentum=model_config.getfloat("momentum"), weight_decay=model_config.getfloat("weight_decay"))

    if basic_config.getboolean("resume"):
        model, optimizer, start_epoch, loss  = load_model(network, checkpoint_path, device, optimizer, True)
    else:
        start_epoch = 0

    writer = SummaryWriter(tensorboard_logdir)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=model_config.getfloat("lr_decay"), patience=model_config.getint("patience"))

    criterion = DiceLoss(classes=n_classes)

    args = TrainerArgs(model_config.getint("n_epochs"), device, model_config.get("model_path"))
    trainer = Trainer(args, network, optimizer, criterion, start_epoch, train_loader, val_loader, scheduler, writer)
    trainer.start()
    print("Finished!")