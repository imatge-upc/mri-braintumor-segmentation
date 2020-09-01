import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F



def passthrough(x, **kwargs):
    return x


def define_non_linearity(non_linearity, nchan):
    if non_linearity == "elu":
        return nn.ELU(inplace=True)
    elif non_linearity == "prelu":
        return nn.PReLU(nchan)
    elif non_linearity == "leaky":
        return nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    elif non_linearity == "relu":
        return nn.ReLU(nchan)
    else:
        return nn.ReLU(nchan)

def normalization(num_channels, typee):
    if typee == "instance":
        return torch.nn.InstanceNorm3d(num_channels)
    elif typee == "group":
        return torch.nn.GroupNorm(2,num_channels)
    else:
        return  torch.nn.BatchNorm3d(num_channels)


class LUConv(nn.Module):
    def __init__(self, nchan, elu, kernel_size, padding):
        super(LUConv, self).__init__()
        self.relu1 = define_non_linearity(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.InstanceNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, non_linearity, kernel_size, padding):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, non_linearity, kernel_size, padding))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, num_features, non_linearity, kernel_size, padding=1):
        super(InputTransition, self).__init__()
        self.num_features = num_features
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=kernel_size, padding=padding)

        self.bn1 = torch.nn.InstanceNorm3d(self.num_features)

        self.relu1 = define_non_linearity(non_linearity, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, non_linearity, kernel_size, padding, dropout=False, larger=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.InstanceNorm3d(outChans)

        self.non_linearity_down_block = define_non_linearity(non_linearity, outChans)
        self.non_linearity_cat_residual = define_non_linearity(non_linearity, outChans)
        self.non_linearity_cat_residual_2 = define_non_linearity(non_linearity, outChans)
        self.larger = larger

        self.do1 = nn.Dropout3d() if dropout else passthrough
        self.convolution_blocks = _make_nConv(outChans, nConvs, non_linearity, kernel_size, padding)
        self.convolution_blocks_2 = _make_nConv(outChans, nConvs, non_linearity, kernel_size, padding)


    def forward(self, x):
        down = self.non_linearity_down_block(self.bn1(self.down_conv(x)))
        out = self.do1(down)

        # convolutions
        out = self.convolution_blocks(out)
        out = self.non_linearity_cat_residual(torch.add(out, down))

        if self.larger:
            res = out
            out = self.convolution_blocks_2(out)
            out = self.non_linearity_cat_residual_2(torch.add(out, res))

        return out


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, non_linearity, kernel_size, padding, dropout=False):
        super(UpTransition, self).__init__()

        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = torch.nn.InstanceNorm3d(outChans // 2)
        self.dropout_skip_connection = nn.Dropout3d()
        self.non_linearity_up_block = define_non_linearity(non_linearity, outChans // 2)
        self.non_linearity_cat_residual = define_non_linearity(non_linearity, outChans)

        self.do1 = nn.Dropout3d() if dropout else passthrough

        self.convolution_blocks = _make_nConv(outChans, nConvs, non_linearity, kernel_size, padding)
        self.conv2 = nn.Conv3d(outChans+4, outChans, kernel_size=1)

    def forward(self, x, skipx, input_mod=None):
        out = self.do1(x)
        out = self.non_linearity_up_block(self.bn1(self.up_conv(out)))

        skipxdo = self.dropout_skip_connection(skipx)
        xcat = torch.cat((out, skipxdo), 1)

        if input_mod is not None:
            xcat_tmp = torch.cat((xcat, input_mod), 1)
            xcat = self.conv2(xcat_tmp)

        out = self.convolution_blocks(xcat)
        out = self.non_linearity_cat_residual(torch.add(out, xcat))
        return out



class OutputTransition(nn.Module):

    def __init__(self, in_channels, classes, non_linearity, kernel_size, padding=1):
        super(OutputTransition, self).__init__()

        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.InstanceNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = define_non_linearity(non_linearity, classes)

        self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        out_scores = out
        # make channels the last axis
        out_scores = out_scores.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out_scores = out_scores.view(out.numel() //  self.classes, self.classes)
        out_scores = self.softmax(out_scores, dim=1)

        return out, out_scores


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, non_linearity="elu", in_channels=1, classes=4, init_features_maps=16, kernel_size=5, padding=2):
        # input channels: the four modalities
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr      = InputTransition(in_channels, init_features_maps, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding)
        self.down_tr32  = DownTransition(init_features_maps, nConvs=1, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.down_tr64  = DownTransition(init_features_maps*2, nConvs=2, non_linearity=non_linearity, kernel_size=kernel_size,padding=padding, dropout=True)
        self.down_tr128 = DownTransition(init_features_maps*4, nConvs=3, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.down_tr256 = DownTransition(init_features_maps*8, nConvs=2, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=False, larger=True)
        self.up_tr256   = UpTransition(init_features_maps*16, init_features_maps*16, nConvs=2, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.up_tr128   = UpTransition(init_features_maps*16, init_features_maps*8, nConvs=2, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.up_tr64    = UpTransition(init_features_maps*8, init_features_maps*4, nConvs=1, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.up_tr32    = UpTransition(init_features_maps*4, init_features_maps*2, nConvs=1, non_linearity=non_linearity, kernel_size=kernel_size, padding=padding, dropout=True)
        self.out_tr     = OutputTransition(init_features_maps*2, classes, non_linearity, kernel_size, padding=padding)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16, x) # add modalities at the last step, concatenated
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        size = 32
        input_tensor = torch.rand(1, self.in_channels, size, size, size)
        ideal_out = torch.rand(1, self.classes, size, size, size)
        out_pred = self.forward(input_tensor)
        assert ideal_out.shape == out_pred.shape
        summary(self.to(torch.device(device)), (self.in_channels, size, size, size), device=device)
        print("Vnet test is complete")


if __name__ == "__main__":
    vnet = VNet(non_linearity="elu", in_channels=1, classes=4, init_features_maps=16, kernel_size=3, padding=1)
    vnet.test()