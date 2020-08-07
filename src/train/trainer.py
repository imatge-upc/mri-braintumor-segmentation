import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset.utils.visualization import plot_batch
from src.models.io_model import save_checkpoint, save_model
from src.metrics.training_metrics import AverageMeter
from src.logging_conf import logger

class TrainerArgs:
    def __init__(self, n_epochs=50, device="cpu", output_path="", loss="dice"):
        self.n_epochs = n_epochs
        self.device = device
        self.output_path = output_path
        self.loss = loss

class Trainer:

    def __init__(self, args, model, optimizer, criterion, start_epoch, train_loader, val_loader, lr_scheduler, writer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_data_loader = train_loader
        self.number_train_data = len(self.train_data_loader)

        self.valid_data_loader = val_loader
        self.number_val_data = len(self.valid_data_loader)

        self.lr_scheduler = lr_scheduler
        self.writer = writer

        self.start_epoch = start_epoch
        self.args = args

    def start(self):
        best_loss = 1000
        val_dice_score = 0

        for epoch in range(self.start_epoch, self.args.n_epochs):
            train_dice_loss, train_dice_score, train_combined_loss, train_ce_loss = self.train_epoch(epoch)
            val_dice_loss, val_dice_score, val_combined_loss, val_ce_loss = self.val_epoch(epoch)

            val_loss = val_dice_loss if self.args.loss == "dice" else val_combined_loss
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            self._epoch_summary(epoch, train_dice_loss, val_dice_loss, train_dice_score, val_dice_score, train_combined_loss, train_ce_loss, val_combined_loss, val_ce_loss)

            is_best = bool(val_loss < best_loss)
            best_loss = val_loss if is_best else best_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': best_loss,
                'val_dice_score': val_dice_score
            }, is_best, self.args.output_path)

        save_model({
                'epoch': self.args.n_epochs + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': best_loss,
                'val_dice_score': val_dice_score
            }, self.args.output_path)


    def train_epoch(self, epoch):

        self.model.train()
        dice_loss_global, ce_loss_global, combined_loss_global = AverageMeter(),  AverageMeter(), AverageMeter()
        dice_score = AverageMeter()

        i = 0
        for data_batch, labels_batch in tqdm(self.train_data_loader, desc="Training epoch"):
            def step(trainer):
                trainer.optimizer.zero_grad()

                inputs = data_batch.float().to(trainer.args.device)
                targets = labels_batch.float().to(trainer.args.device)
                inputs.require_grad = True

                predictions, _ = trainer.model(inputs)

                if trainer.args.loss == "dice":
                    dice_loss, mean_dice, subregion_loss = trainer.criterion(predictions, targets)
                    dice_loss.backward()
                    trainer.optimizer.step()

                    print(subregion_loss, dice_loss)
                    if subregion_loss:
                        trainer.writer.add_scalar('Training Dice Loss WT', subregion_loss[0].detach().item(),
                                                  epoch * trainer.number_train_data + i)
                        trainer.writer.add_scalar('Training Dice Loss TC', subregion_loss[1].detach().item(),
                                                  epoch * trainer.number_train_data + i)
                        trainer.writer.add_scalar('Training Dice Loss ET', subregion_loss[2].detach().item(),
                                                  epoch * trainer.number_train_data + i)

                else:
                    combined_loss, dice_loss, ce_loss, mean_dice = trainer.criterion(predictions, targets)
                    combined_loss.backward()
                    trainer.optimizer.step()

                    combined_loss = combined_loss.detach().item()
                    ce_loss = ce_loss.detach().item()

                    combined_loss_global.update(combined_loss, data_batch.size(0))
                    ce_loss_global.update(ce_loss, data_batch.size(0))

                    trainer.writer.add_scalar('Train combined CE-Dice Loss', combined_loss, epoch * trainer.number_train_data + i)
                    trainer.writer.add_scalar('Train Cross Entropy Loss', ce_loss, epoch * trainer.number_train_data + i)


                dice_loss = dice_loss.detach().item()
                mean_dice = mean_dice.detach().item()
                dice_loss_global.update(dice_loss, data_batch.size(0))
                dice_score.update(mean_dice, data_batch.size(0))


                trainer.writer.add_scalar('Training Dice Loss', dice_loss, epoch * trainer.number_train_data + i)
                trainer.writer.add_scalar('Training Dice Score', mean_dice, epoch * trainer.number_train_data + i)

                trainer._add_image(data_batch, False, "Modality patch")
                trainer._add_image(labels_batch, True, "Segmentation ground truth patch")
                trainer._add_image(predictions.max(1)[1], True, "Segmentation prediction patch")


            step(self)


            i += 1
        if self.args.loss == "dice":
            return dice_loss_global.avg(), dice_score.avg(), 0, 0
        else:
            return dice_loss_global.avg(), dice_score.avg(), combined_loss_global.avg(), ce_loss_global.avg()


    def _add_image(self, batch, seg=False, title=""):
        plot_buf = plot_batch(batch, seg=seg, slice=32, batch_size=len(batch))
        im = Image.open(plot_buf)
        image = T.ToTensor()(im)
        self.writer.add_image(title, image)


    def val_epoch(self, epoch):
        self.model.eval()
        losses, ce_loss_global, combined_loss_global = AverageMeter(), AverageMeter(), AverageMeter()
        dice_score = AverageMeter()

        i = 0
        for data_batch, labels_batch in tqdm(self.valid_data_loader, desc="Validation epoch"):

            def step(trainer):
                inputs = data_batch.float().to(trainer.args.device)
                targets = labels_batch.float().to(trainer.args.device)

                with torch.no_grad():
                    outputs, _ = trainer.model(inputs)

                    if trainer.args.loss == "dice":
                        loss_dice, mean_dice, subregion_loss = trainer.criterion(outputs, targets)

                        if subregion_loss:
                            trainer.writer.add_scalar('Training Dice Loss WT', subregion_loss[0].detach().item(),
                                                      epoch * trainer.number_train_data + i)
                            trainer.writer.add_scalar('Training Dice Loss TC', subregion_loss[1].detach().item(),
                                                      epoch * trainer.number_train_data + i)
                            trainer.writer.add_scalar('Training Dice Loss ET', subregion_loss[2].detach().item(),
                                                      epoch * trainer.number_train_data + i)

                    else:
                        combined_loss, loss_dice, ce_loss, mean_dice = trainer.criterion(outputs, targets)
                        combined_loss = combined_loss.detach().item()
                        ce_loss = ce_loss.detach().item()
                        combined_loss_global.update(combined_loss, data_batch.size(0))
                        ce_loss_global.update(ce_loss, data_batch.size(0))

                        trainer.writer.add_scalar('Validation Combined CE-Dice Loss', combined_loss,epoch * trainer.number_val_data + i)
                        trainer.writer.add_scalar('Validation Cross Entropy Loss', ce_loss, epoch * trainer.number_val_data + i)

                    loss_dice = loss_dice.detach().item()
                    mean_dice = mean_dice.detach().item()
                    losses.update(loss_dice, data_batch.size(0))
                    dice_score.update(mean_dice, data_batch.size(0))

                trainer.writer.add_scalar('Validation Dice Loss', loss_dice, epoch * trainer.number_val_data + i)
                trainer.writer.add_scalar('Validation Dice Score', mean_dice, epoch * trainer.number_val_data + i)

                trainer._add_image(data_batch, False, "Val Modality patch")
                trainer._add_image(labels_batch, True, "Val Segmentation ground truth patch")
                trainer._add_image(outputs.max(1)[1], True, "Val Segmentation prediction patch")

            step(self)

            i += 1

        if  self.args.loss == "dice":
            return losses.avg(), dice_score.avg(), 0, 0
        else:
            return losses.avg(), dice_score.avg(), combined_loss_global.avg(), ce_loss_global.avg()

    def _epoch_summary(self, epoch, train_loss, val_loss, train_dice_score, val_dice_score, train_combined_loss, train_ce_loss, val_combined_loss, val_ce_loss):

        if self.args.loss == "dice" or  self.args.loss == "dice_eval":
            logger.info(f'epoch: {epoch}\n '
                        f'** Dice Loss **  : train_loss: {train_loss:.2f} | val_loss {val_loss:.2f} \n'
                        f'** Dice Score ** : train_dice_score {train_dice_score:.2f} | val_dice_score {val_dice_score:.2f}')

        else:
            logger.info(f'epoch: {epoch}\n'
                        f'** Combined Loss **  : train_loss: {train_combined_loss:.2f} | val_loss {val_combined_loss:.2f} \n'
                        f'** CE Loss **        : train_loss {train_ce_loss:.2f} | val_loss {val_ce_loss:.2f}\n'
                        f'** Dice Loss **      : train_loss: {train_loss:.2f} | val_loss {val_loss:.2f} \n'
                        f'** Dice Score **     : train_dice_score {train_dice_score:.2f} | val_dice_score {val_dice_score:.2f}\n'
                        )


