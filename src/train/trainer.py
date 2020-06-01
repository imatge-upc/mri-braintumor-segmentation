from src.models.io_model import save_checkpoint
from tqdm import tqdm

from src.metrics.training_metrics import AverageMeter
from src.logging_conf import logger



class TrainerArgs:
    def __init__(self, n_epochs=50, device="cpu", output_path=""):
        self.n_epochs = n_epochs
        self.device = device
        self.output_path = output_path

class Trainer:

    def __init__(self, args, model, optimizer, criterion, train_loader, val_loader, lr_scheduler, writer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_loader
        self.len_epoch = len(self.train_data_loader)

        self.valid_data_loader = val_loader

        self.lr_scheduler = lr_scheduler
        self.writer = writer

        self.start_epoch = 0
        self.args = args

    def start(self):
        best_loss = 1000

        for epoch in range(self.start_epoch, self.args.n_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            self._epoch_summary(epoch, train_loss, val_loss)
            is_best = bool(val_loss < best_loss)
            best_loss = val_loss if is_best else best_loss
            save_checkpoint({
                'epoch': self.start_epoch + epoch + 1,
                'state_dict': self.model.state_dict(),
                'val_loss': best_loss
            }, is_best, self.args.output_path)


    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        i = 0
        for patients_ids, data_batch, labels_batch in tqdm(self.train_data_loader, desc="Training epoch"):

            self.optimizer.zero_grad()

            inputs = data_batch.float().to(self.args.device)
            targets = labels_batch.float().to(self.args.device)
            inputs.require_grad = True

            outputs = self.model(inputs)

            loss_dice, per_ch_score = self.criterion(outputs, targets)

            loss_dice.backward()
            self.optimizer.step()

            losses.update(loss_dice.cpu(), data_batch.size(0))
            self.writer.add_scalar('Training loss', loss_dice.item(), epoch * len(self.train_data_loader) + i)
            i += 1

        return losses.avg

    def val_epoch(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        i = 0
        for patients_ids, data_batch, labels_batch in tqdm(self.valid_data_loader, desc="Validation epoch"):

            inputs = data_batch.float().to(self.args.device)
            targets = labels_batch.float().to(self.args.device)
            inputs.require_grad = False

            outputs = self.model(inputs)

            loss_dice, per_ch_score = self.criterion(outputs, targets)

            loss_dice.backward()

            losses.update(loss_dice.cpu(), data_batch.size(0))
            self.writer.add_scalar('Validation loss', loss_dice.item(), epoch * len(self.valid_data_loader) + i)
            i += 1

        return losses.avg

    def _epoch_summary(self, epoch, train_loss, val_loss):
        logger.info(
            f'epoch: {epoch} | train_loss: {train_loss:.2f} | val_loss {val_loss:.2f}')

