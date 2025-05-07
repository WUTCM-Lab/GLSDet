import torch
import torch.nn as nn
from models.base.darknet import BaseConv


class GridConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=512, stride=1, act="silu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.top_conv = BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=stride, act=act)
        self.bottom_conv = BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=5, stride=stride, act=act)
        self.channel_conv = nn.Conv2d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        feat_grid_top = x[:, :, :int(x.shape[3] / 2), :]
        feat_grid_bottom = x[:, :, int(x.shape[3] / 2):, :]

        feat_grid_top = self.top_conv(feat_grid_top)
        feat_grid_bottom = self.bottom_conv(feat_grid_bottom)

        outputs = torch.cat([feat_grid_top, feat_grid_bottom], dim=2)
        outputs = self.channel_conv(outputs)

        return outputs
