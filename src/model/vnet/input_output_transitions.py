import torch
import torch.nn as nn
import torch.nn.functional as F

from .cont_batch_norm_3d import ContBatchNorm3d
from .activation_functions import add_elu


class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels, elu):
        super(InputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=5, padding=2)

        self.bn1 = ContBatchNorm3d(out_channels)
        self.relu1 = add_elu(elu, out_channels)

    def forward(self, x):

        conv_x = self.conv1(x)
        out = self.bn1(conv_x)
        join = torch.add(out, x)
        out = self.relu1(join)

        # x16 = torch.cat( (x,)*n_channels_to_split, 0) (inside the add)

        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = add_elu(elu, 2)

        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)

        return out
