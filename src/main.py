import importlib
import sys
import torch
from torchvision import transforms


from src.dataset.train_val_split import train_val_split
from src.losses.ce_dice_loss import CrossEntropyDiceLoss3D

from src.losses import dice_loss, region_based_loss
from src.models.io_model import load_model
from src.train.trainer import Trainer, TrainerArgs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.config import BratsConfiguration
from src.dataset.loaders.brats_dataset import BratsDataset
from src.dataset.augmentations import brats_augmentations

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
loss = model_config.get("loss")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")


######## DATASET
logger.info("Creating Dataset...")

data, _ = dataset.read_brats(dataset_config.get("train_csv"), lgg_only=dataset_config.getboolean("lgg_only"))
data_train, data_val = train_val_split(data, val_size=0.1)
data_train = data_train * n_patches
data_val = data_val * n_patches

n_modalities = dataset_config.getint("n_modalities") # like color channels

sampling_method = importlib.import_module(dataset_config.get("sampling_method"))


transform = transforms.Compose([brats_augmentations.RandomIntensityShift(),
                                brats_augmentations.RandomIntensityScale(),
                                brats_augmentations.RandomMirrorFlip(p=0.5)])


compute_patch = basic_config.getboolean("compute_patches")
train_dataset = BratsDataset(data_train, sampling_method, patch_size, compute_patch=compute_patch, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = BratsDataset(data_val, sampling_method, patch_size, compute_patch=compute_patch, transform=None)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if basic_config.getboolean("plot"):
    x, y = next(iter(train_loader))
    print(x.shape)
    logger.info('Plotting images')
    visualization.plot_batch_slice(x, y, slice=30, save=True)


######## MODEL
logger.info("Initiating Model...")

if model_config["network"] == "vnet":

    network = vnet.VNet(elu=model_config.getboolean("use_elu"),
                        in_channels=n_modalities,
                        classes=n_classes,
                        init_features_maps=model_config.getint("init_features_maps"))

    n_params = sum([p.data.nelement() for p in network.parameters()])
    logger.info("Number of params: {}".format(n_params))
else:
    raise ValueError("Bad parameter for network {}".format(model_config.get("network")))



if basic_config.getboolean("train_flag"):
    ##### TRAIN
    logger.info("Start Training")
    network.to(device)

    optim = model_config.get("optimizer")

    if optim == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), lr=model_config.getfloat("learning_rate"),
                                    momentum=model_config.getfloat("momentum"), weight_decay=model_config.getfloat("weight_decay"))
    elif optim == "ADAM":
        optimizer = torch.optim.Adam(network.parameters(), lr=model_config.getfloat("learning_rate"),
                         weight_decay=model_config.getfloat("weight_decay"), amsgrad=False)
    else:
        raise ValueError("Bad optimizer. Current options: [SGD, ADAM]")


    if basic_config.getboolean("resume"):
        logger.info("Loading model from checkpoint..")
        model, optimizer, start_epoch ,_  = load_model(network, checkpoint_path, device, optimizer, True)
        logger.info(f"Loaded model with starting epoch {start_epoch}")
    else:
        start_epoch = 0


    writer = SummaryWriter(tensorboard_logdir)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=model_config.getfloat("lr_decay"), patience=model_config.getint("patience"))

    if loss == "dice":
        criterion = dice_loss.DiceLoss(classes=n_classes, eval_regions=model_config.getboolean("eval_regions"),
                                       sigmoid_normalization=True)

    elif loss == "combined":
        # 0. back, 1: ncr, 2: ed, 3: et
        ce_weigh = torch.tensor([0.1, 0.35, 0.2 , 0.35])
        criterion = CrossEntropyDiceLoss3D(weight=ce_weigh, classes=n_classes,
                                           eval_regions=model_config.getboolean("eval_regions"), sigmoid_normalization=True)
    elif loss == "both_dice":
        criterion = region_based_loss.RegionBasedDiceLoss3D(classes=n_classes, sigmoid_normalization=True)

    else:
        raise ValueError(f"Bad loss value {loss}. Expected ['dice', combined]")

    args = TrainerArgs(model_config.getint("n_epochs"), device, model_config.get("model_path"), loss)
    trainer = Trainer(args, network, optimizer, criterion, start_epoch, train_loader, val_loader, scheduler, writer)
    trainer.start()
    print("Finished!")