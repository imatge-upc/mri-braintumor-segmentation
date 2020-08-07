"""
Code was adapted and modified from https://github.com/black0017/MedicalZooPytorch
"""
from typing import Tuple
import torch
from torch import nn



class DiceLoss(nn.Module):
    """
    Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """
    def __init__(self, classes=4, weight=None, sigmoid_normalization=True, eval_regions: bool=False):
        super(DiceLoss, self).__init__()

        self.register_buffer('weight', weight)
        self.normalization = nn.Sigmoid() if sigmoid_normalization else nn.Softmax(dim=1)
        self.classes = classes
        self.eval_regions = eval_regions

    def _flatten(self, tensor: torch.tensor) -> torch.tensor:
        """
        Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1) # number of channels
        axis_order = (1, 0) + tuple(range(2, tensor.dim())) # new axis order
        transposed = tensor.permute(axis_order) # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        return transposed.contiguous().view(C, -1) # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)

    def dice(self, input: torch.tensor, target: torch.tensor, weight: float, epsilon=1e-6) -> float:
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

        :param input: NxCxSpatial input tensor
        :param target:  NxCxSpatial target tensor
        :param weight: Cx1 tensor of weight per channel/class
        :param epsilon: prevents division by zero
        :return: dice loss, dice score

        """
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self._flatten(input)
        target = self._flatten(target)
        target = target.float()

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        union = (input * input).sum(-1) + (target * target).sum(-1)
        return 2 * (intersect / union.clamp(min=epsilon))


    def reformat_labels(self, seg_mask):
        et = seg_mask[..., 3]
        tc = seg_mask[..., 3] + seg_mask[..., 1]
        wt = seg_mask[..., 3] + seg_mask[..., 1] + seg_mask[..., 2]
        return torch.stack([seg_mask[..., 0], wt, tc, et], dim=-1)

    def forward(self, input: torch.tensor, target: torch.tensor) -> Tuple[float, float]:

        target = torch.nn.functional.one_hot(target.long(), self.classes)

        assert input.dim() == target.dim() == 5, f"'input' {input.dim()} and 'target' {target.dim()} have different number of dims "

        input = self.normalization(input.float())
        if self.eval_regions:
            input = self.reformat_labels(input.permute((0,2,3,4,1)))
            target = self.reformat_labels(target)

        per_channel_dice = self.dice(input, target, weight=self.weight) # compute per channel Dice coefficient
        mean = torch.mean(per_channel_dice)
        loss = (1. - mean)

        # average Dice score across all channels/classes
        return loss, mean