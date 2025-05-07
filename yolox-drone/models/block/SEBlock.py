import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()
        y1 = self.avg_pool(X_input)
        y2 = self.max_pool(X_input)
        z = [y1, y2]
        w = []
        for y in z:
            y = y.view(b, c)
            y = self.linear1(y)
            w.append(y)

        y = w[0] + w[1]
        y = self.linear2(y)
        y = y.view(b, c, 1, 1)
        return X_input * y.expand_as(X_input)


class SEBlockFPN(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlockFPN, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.reduction = reduction
        self.linear1 = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        m = []
        inputs_channels = []
        for feature in inputs:
            tmp = self.avg_pool(feature)
            m.append(tmp)
            inputs_channels.append(feature.shape[1])
        m = torch.cat(m, dim=1)

        b, c, _, _ = m.size()
        y = m.view(b, c)
        y = self.linear1(y)
        y = self.linear2(y)
        y = y.view(b, c, 1, 1)

        outputs = []
        cur = 0
        for k, feature in enumerate(inputs):
            outputs.append(feature * y[:, cur: cur + inputs_channels[k], :, :])
            cur += inputs_channels[k]

        return outputs


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEAttention, self).__init__()
        self.reduction = reduction
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        m = inputs.view(b, c, h*w).permute(0, 2, 1)
        m = torch.unsqueeze(m, dim=2)
        n = self.avg_pool(m).view(b, h*w)
        n = self.linear1(n)
        n = self.linear2(n)
        n = n.view(b, h*w, 1, 1)
        outputs = m * n
        return outputs.reshape(b, h, w, c).permute(0, 3, 1, 2)
