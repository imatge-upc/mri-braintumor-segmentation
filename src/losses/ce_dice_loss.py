from torch import  nn
import torch

from src.losses.dice_loss import DiceLoss



class CrossEntropyDiceLoss3D(nn.Module):

    def __init__(self, weight, classes):
        super(CrossEntropyDiceLoss3D, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = DiceLoss(classes=classes)

    def forward(self, input: torch.tensor, target: torch.tensor, weight: torch.tensor=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxDxHxW) Network output
        :param target: ground truth torch.tensor (NxDxHxW)
        :param weight: torch.tensor (N) to provide class weights
        :return: scalar
       """
        dice_loss, dice_score = self.dice_loss(input, target)

        ce_loss = self.cross_entropy_loss(input, target.long())

        total_loss = dice_loss + ce_loss

        return total_loss, dice_loss, ce_loss, dice_score