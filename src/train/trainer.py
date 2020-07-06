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
            train_dice_loss, train_dice_score = self.train_epoch(epoch)
            val_dice_loss, val_dice_score = self.val_epoch(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step(val_dice_loss)

            self._epoch_summary(epoch, train_dice_loss, val_dice_loss, train_dice_score, val_dice_score)
            is_best = bool(val_dice_loss < best_loss)
            best_loss = val_dice_loss if is_best else best_loss
            save_checkpoint({
                'epoch': self.start_epoch + epoch + 1,
                'state_dict': self.model.state_dict(),
                'val_loss': best_loss,
                'val_dice_score': val_dice_score
            }, is_best, self.args.output_path)


    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        dice_score = AverageMeter()

        i = 0
        for patients_ids, data_batch, labels_batch in tqdm(self.train_data_loader, desc="Training epoch"):

            self.optimizer.zero_grad()

            inputs = data_batch.float().to(self.args.device)
            targets = labels_batch.float().to(self.args.device)
            inputs.require_grad = True

            outputs = self.model(inputs)

            loss_dice, mean_dice = self.criterion(outputs, targets)

            loss_dice.backward()
            self.optimizer.step()

            losses.update(loss_dice.cpu(), data_batch.size(0))
            dice_score.update(mean_dice.cpu(), data_batch.size(0))

            self.writer.add_scalar('Training Dice Loss', loss_dice.item(), epoch * len(self.train_data_loader) + i)
            self.writer.add_scalar('Training Dice Score', mean_dice.item(), epoch * len(self.train_data_loader) + i)

            i += 1

        return losses.avg, dice_score.avg

    def val_epoch(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        dice_score = AverageMeter()
        i = 0
        for patients_ids, data_batch, labels_batch in tqdm(self.valid_data_loader, desc="Validation epoch"):

            inputs = data_batch.float().to(self.args.device)
            targets = labels_batch.float().to(self.args.device)
            inputs.require_grad = False

            outputs = self.model(inputs)

            loss_dice, mean_dice = self.criterion(outputs, targets)

            loss_dice.backward()

            losses.update(loss_dice.cpu(), data_batch.size(0))
            dice_score.update(mean_dice.cpu(), data_batch.size(0))
            self.writer.add_scalar('Validation Dice Loss', loss_dice.item(), epoch * len(self.valid_data_loader) + i)
            self.writer.add_scalar('Validation Dice Score', mean_dice.item(), epoch * len(self.valid_data_loader) + i)
            i += 1

        return losses.avg, dice_score.avg

    def _epoch_summary(self, epoch, train_loss, val_loss, train_dice_score, val_dice_score):
        logger.info(f'epoch: {epoch}\n '
                    f'**Dice Loss: train_loss: {train_loss:.2f} | val_loss {val_loss:.2f} \n'
                    f'**Dice Score: train_dice_score {train_dice_score:.2f} | val_dice_score {val_dice_score:.2f}')

