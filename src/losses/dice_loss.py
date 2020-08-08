"""
Code was adapted and modified from https://github.com/black0017/MedicalZooPytorch
"""
from typing import Tuple
import torch
from torch import nn

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)



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

    def _reformat_labels(self, seg_mask):
        wt = torch.stack([ seg_mask[..., 0], torch.sum(seg_mask[..., [1, 2, 3]], dim=-1)])
        tc = torch.stack([ seg_mask[..., 0], torch.sum(seg_mask[..., [1, 3]], dim=-1)])
        et = torch.stack([ seg_mask[..., 0], seg_mask[..., 3]])
        return wt, tc, et

    def dice(self, input: torch.tensor, target: torch.tensor, weight: float, epsilon=1e-6) -> float:
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

        :param input: NxCxSpatial input tensor
        :param target:  NxCxSpatial target tensor
        :param weight: Cx1 tensor of weight per channel. Channels represent the class
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


    def forward(self, input: torch.tensor, target: torch.tensor) -> Tuple[float, float, list]:

        target = torch.nn.functional.one_hot(target.long(), self.classes)

        assert input.dim() == target.dim() == 5, f"'input' {input.dim()} and 'target' {target.dim()} have different number of dims "

        input = self.normalization(input.float())
        input = input.permute((0, 2, 3, 4, 1))

        if self.eval_regions:
            input_wt, input_tc, input_et = self._reformat_labels(input)
            target_wt, target_tc, target_et = self._reformat_labels(target)

            wt_dice = torch.mean(self.dice(input_wt, target_wt, weight=self.weight))
            tc_dice = torch.mean(self.dice(input_tc, target_tc, weight=self.weight))
            et_dice = torch.mean(self.dice(input_et, target_et, weight=self.weight))

            wt_loss = 1 - wt_dice
            tc_loss = 1 - tc_dice
            et_loss = 1 - et_dice

            loss = 1/3 * (wt_loss + tc_loss + et_loss)
            score = 1/3 * (wt_dice + tc_dice + et_dice)

            return loss, score, [wt_loss, tc_loss, et_loss]

        else:
            per_channel_dice = self.dice(input, target, weight=self.weight) # compute per channel Dice coefficient
            mean = torch.mean(per_channel_dice)
            loss = (1. - mean)
            # average Dice score across all channels/classes
            return loss, mean, []