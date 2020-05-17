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
    '''
    Decoder output layer
    output the prediction of segmentation result
    '''

    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = nn.Sigmoid()

    def forward(self, x):
        return self.actv1(self.conv1(x))