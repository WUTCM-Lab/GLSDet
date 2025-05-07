import torch
from torch import nn
from models.base.baseConv import BaseConv as Conv2d

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FFA(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = Conv2d(
                num_channels * 2,
                num_channels * 4,
                ksize=1,
                stride=1,
                act='relu'
            )

        self.create_content_extractor = nn.Sequential(*[
            Conv2d(
                num_channels * 4,
                num_channels * 4,
                ksize=1,
                stride=1,
                act='relu'
            ),
            Conv2d(
                num_channels * 4,
                num_channels * 4,
                ksize=1,
                stride=1,
                act='relu'
            )
        ])

        self.create_text_extractor = nn.Sequential(*[
            Conv2d(
                num_channels * 2,
                num_channels * 2,
                ksize=1,
                stride=1,
                act='relu'
            )
        ])

        self.conv3 =  Conv2d(
                num_channels * 2,
                num_channels,
                ksize=1,
                stride=1,
                act='relu'
            )

        self.subpixel = nn.PixelShuffle(2)

        self.se1 = SE(num_channels * 4)
        # self.se2 = SE(num_channels * 2)


    def forward(self, bottom, top):
        top_processed = self.scale(top)
        top_processed = self.create_content_extractor(top_processed)
        top_processed =top_processed + self.se1(top_processed)
        top_processed = self.subpixel(top_processed)
        bottom_processed = torch.cat((bottom, top_processed), 1)
        bottom_processed = self.create_text_extractor(bottom_processed)
        # bottom_processed = bottom_processed + self.se2(bottom_processed)
        bottom_processed = self.conv3(bottom_processed)
        out = top_processed + bottom_processed

        return out







