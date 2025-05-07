import torch
import torch.nn.functional as F
from torch import nn
from .dyrelu import h_sigmoid
import collections
from itertools import repeat
from models.base.activation import get_activation
from models.base.baseConv import BaseConv
from models.dyhead.dyrelu import DYReLU


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = BaseConv(in_channels, out_channels, ksize=3, stride=stride)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input):
        x = self.conv(input.contiguous())
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = get_activation('silu', inplace=True)
        self.init_weights()
        # self.relu2 = DYReLU(in_channels, out_channels)

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            temp_fea = [self.DyConv[1](feature)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]]))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]]),
                                                    size=[feature.size(2), feature.size(3)]))
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            # f,n, c,h,w
            tt_fea = (res_fea * spa_pyr_attn).permute(1, 2, 0, 3, 4)
            func_pool = nn.MaxPool3d(kernel_size=(tt_fea.shape[2], 3, 3), padding=(0, 1, 1), stride=1)
            final_fea = func_pool(tt_fea).squeeze(dim=2)

            next_x[name] = final_fea

        return next_x


class DyHead(nn.Module):
    def __init__(self, blocks_num=6, in_channels=256):
        super(DyHead, self).__init__()
        dyhead = []
        for i in range(blocks_num):
            dyhead.append(
                DyConv(
                    in_channels,
                    in_channels,
                    conv_func=Conv3x3Norm,
                )
            )
        self.dyhead = nn.Sequential(*dyhead)

    def forward(self, x):
        outputs = self.dyhead(x)
        return outputs
