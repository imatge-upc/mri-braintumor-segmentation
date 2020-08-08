from torch import  nn
import torch

from src.losses.dice_loss import DiceLoss



class CrossEntropyDiceLoss3D(nn.Module):

    def __init__(self, weight: torch.tensor, classes: int, eval_regions: bool=True, sigmoid_normalization: bool=True):

        super(CrossEntropyDiceLoss3D, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)

        self.dice_loss = DiceLoss(classes=classes, sigmoid_normalization=sigmoid_normalization, eval_regions=eval_regions)


    def forward(self, input: torch.tensor, target: torch.tensor, weight_ce: int=1, weight_dice: int=1):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.

        Forward pass

        :param input: torch.tensor (NxCxDxHxW) Network output
        :param target: ground truth torch.tensor (NxDxHxW)
        :param weight: torch.tensor (N) to provide class weights
        :return: scalar
       """
        dice_loss, dice_score, subregions = self.dice_loss(input, target)

        ce_loss = self.cross_entropy_loss(input, target.long())

        total_loss = weight_dice*dice_loss + weight_ce*ce_loss

        return total_loss, dice_loss, ce_loss, dice_score, subregions