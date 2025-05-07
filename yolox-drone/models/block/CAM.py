import torch
import torch.nn as nn


class ConAugModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)

    def forward(self, inputs):
        outputs = torch.cat((self.conv1(inputs),
                             self.conv2(inputs),
                             self.conv3(inputs),
                             ), dim=1)
        return outputs