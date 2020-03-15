'''
Source material from: https://github.com/mattmacy/vnet.pytorch
'''
import torch.nn as nn
from model.vnet.input_output_transitions import InputTransition, OutputTransition
from model.vnet.up_down_transitions import DownTransition, UpTransition


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu: bool=True, nll:bool=False, batch_size: int=1):
        super(VNet, self).__init__()

        self.batch_size = batch_size
        self.in_tr = InputTransition(in_channels=self.batch_size,
                                     out_channels=16, elu=elu)
        # Downsampling
        self.down_tr32 = DownTransition(inChans=16, nConvs=1, elu=elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        # Upsampling
        self.up_tr256 = UpTransition(inChans=256, outChans=256, nConvs=1, elu=elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)

        # Output
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out