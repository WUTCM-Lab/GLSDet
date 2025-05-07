import torch.nn as nn
import torch
import torch.nn.functional as F
from models.base.activation import get_activation
from models.base.baseConv import BaseConv


class FeatureGroup(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels * self.in_channels, kernel_size=1, stride=1)
        self.relu = get_activation('relu')
        self.gn = nn.GroupNorm(num_groups=16, num_channels=self.in_channels)

    def forward(self, inputs):
        reshape_inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        m = F.adaptive_avg_pool2d(inputs, (1, 1))
        m = self.conv1(m)
        m = self.gn(m)
        m = self.relu(m)
        m = self.conv2(m).view((inputs.shape[0], inputs.shape[1], inputs.shape[1]))
        x = torch.matmul(m, reshape_inputs).view(inputs.shape)
        return x


class FeatureGroupFPN(nn.Module):
    def __init__(self, num_group=3, in_channels=192, feature_group=True):
        """
        [b, c , h, w],[b, c , h//2, w//2],[b, c , h//4, w//4]...
        """
        super().__init__()
        self.num_group = num_group
        self.in_channels = in_channels
        # self.conv_list = nn.ModuleList()
        self.span = int(self.in_channels / self.num_group)
        self.feature_group = feature_group
        self.feature_groups = nn.ModuleList()

        if self.feature_group:
            for i in range(self.num_group):
                self.feature_groups.append(FeatureGroup(self.in_channels))

        # for i in range(self.num_group):
        #
        #     for j in range(i, 0, -1):
        #         stride = pow(2, j)
        #         print(stride)
                # self.conv_list.append(nn.Conv2d(self.span, self.span, kernel_size=3, stride=stride, padding=1))

    def forward(self, inputs):
        processed_inputs = []
        if self.feature_group:
            for k, x in enumerate(inputs):
                processed_inputs.append(self.feature_groups[k](x))
        else:
            processed_inputs = inputs

        outputs = []
        k = 0
        span = self.span
        for i in range(self.num_group):
            mix_feature = []
            h = inputs[i].shape[2]
            for x in processed_inputs:
                h_x = x.shape[2]
                if h_x == h:
                    mix_feature.append(x[0:, span * i:span * (i+1), 0:, 0:])
                if h_x > h:
                    # tmp = self.conv_list[k](x[0:, span * i:span * (i+1), 0:, 0:])
                    tmp = F.adaptive_max_pool2d(x[0:, span * i:span * (i+1), 0:, 0:],
                                                (inputs[i].shape[2], inputs[i].shape[3]))
                    k = k+1
                    mix_feature.append(tmp)
                if h_x < h:
                    tmp = F.upsample_bilinear(x[0:, span * i:span * (i+1), 0:, 0:],
                                              size=(inputs[i].shape[2], inputs[i].shape[3]))
                    mix_feature.append(tmp)
            mix_feature = torch.cat(mix_feature, dim=1)
            outputs.append(mix_feature)
        return outputs


class CascadeFeatureGroupFPN(nn.Module):
    def __init__(self, num_stages, num_group=3, in_channels=192, feature_group=True, act='relu'):
        super().__init__()

        self.get_weight = nn.Sequential(*[
            BaseConv(in_channels=in_channels, out_channels=in_channels, ksize=3, stride=1, act=act),
            BaseConv(in_channels=in_channels, out_channels=in_channels, ksize=3, stride=1, act=act),
        ])
        self.fpn_list = nn.ModuleList()
        self.num_group = num_group
        self.num_stages = num_stages
        for _ in range(num_stages):
            self.fpn_list.append(FeatureGroupFPN(num_group, in_channels, feature_group))

    def forward(self, inputs):
        cur_processed = inputs
        final_outputs = []
        for k in range(self.num_stages):
            outputs = self.fpn_list[k](cur_processed)
            cur_processed = outputs
            for i in range(self.num_group):
                if len(final_outputs) == self.num_group:
                    final_outputs[i] += self.get_weight(cur_processed[i]) * cur_processed[i]
                else:
                    final_outputs.append(self.get_weight(cur_processed[i]) * cur_processed[i])
        return final_outputs
