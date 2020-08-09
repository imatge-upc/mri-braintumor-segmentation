import torch
from src.losses.dice_loss import DiceLoss
from torch import nn



class RegionBasedDiceLoss3D(nn.Module):

    def __init__(self, classes: int, sigmoid_normalization: bool=True):

        super(RegionBasedDiceLoss3D, self).__init__()

        self.dice_loss = DiceLoss(classes=classes, sigmoid_normalization=sigmoid_normalization,
                                               eval_regions=False)

        self.dice_loss_region_based = DiceLoss(classes=classes, sigmoid_normalization=sigmoid_normalization,
                                               eval_regions=True)


    def forward(self, input: torch.tensor, target: torch.tensor, weight_reg: int=1, weight_dice: int=1):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.

        Forward pass

        :param input: torch.tensor (NxCxDxHxW) Network output
        :param target: ground truth torch.tensor (NxDxHxW)
        :param weight: torch.tensor (N) to provide class weights
        :return: scalar
       """
        dice_loss, dice_score, _ = self.dice_loss(input, target)
        dice_loss_reg, _, subregions = self.dice_loss_region_based(input, target)

        total_loss = weight_dice*dice_loss + weight_reg*dice_loss_reg

        return total_loss, dice_loss, dice_score, dice_loss_reg, subregions