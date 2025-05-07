import torch
from torch import nn
from torch.nn import functional as F
from models.base.darknet import BaseConv


class Identity_Conv(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups)
        conv_weight = torch.zeros(self.conv.state_dict()['weight'].shape, dtype=torch.float32)
        conv_bias = torch.zeros(self.conv.state_dict()['bias'].shape, dtype=torch.float32)
        num_filters = conv_weight.shape[0]
        if groups == 1:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, idx_filters, :] = 1.0
        else:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, int(idx_filters % (num_filters / groups)), 1, 1] = 1.0
        self.conv.weight.data = conv_weight
        self.conv.bias.data = conv_bias

    def forward(self, x):
        return self.conv(x)


class Identity_Conv_three(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        conv_weight = torch.zeros(self.conv.weight.data.shape, dtype=torch.float32)
        conv_bias = torch.zeros(self.conv.bias.data.shape, dtype=torch.float32)
        num_filters = conv_weight.shape[0]
        if groups == 1:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, idx_filters, 1, 1] = 1.0
        else:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, int(idx_filters % (num_filters / groups)), 1, 1] = 1.0
        self.conv.weight.data = conv_weight
        self.conv.bias.data = conv_bias

    def forward(self, x):
        return self.conv(x)


class Identity_Conv_five(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        conv_weight = torch.zeros(self.conv.state_dict()['weight'].shape, dtype=torch.float32)
        conv_bias = torch.zeros(self.conv.state_dict()['bias'].shape, dtype=torch.float32)
        num_filters = conv_weight.shape[0]
        if groups == 1:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, idx_filters, 2, 2] = 1.0
        else:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, int(idx_filters % (num_filters / groups)), 1, 1] = 1.0
        self.conv.weight.data = conv_weight
        self.conv.bias.data = conv_bias

    def forward(self, x):
        return self.conv(x)


class Identity_Conv_seven(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        conv_weight = torch.zeros(self.conv.weight.data.shape, dtype=torch.float32)
        conv_bias = torch.zeros(self.conv.bias.data.shape, dtype=torch.float32)
        num_filters = conv_weight.shape[0]
        if groups == 1:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, idx_filters, 3, 3] = 1.0
        else:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, int(idx_filters % (num_filters / groups)), 1, 1] = 1.0
        self.conv.weight.data = conv_weight
        self.conv.bias.data = conv_bias

    def forward(self, x):
        return self.conv(x)


class Identity_Conv_nine(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=9, stride=1, padding=4, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        conv_weight = torch.zeros(self.conv.weight.data.shape, dtype=torch.float32)
        conv_bias = torch.zeros(self.conv.bias.data.shape, dtype=torch.float32)
        num_filters = conv_weight.shape[0]
        if groups == 1:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, idx_filters, 4, 4] = 1.0
        else:
            for idx_filters in range(num_filters):
                conv_weight[idx_filters, int(idx_filters % (num_filters / groups)), 1, 1] = 1.0
        self.conv.weight.data = conv_weight
        self.conv.bias.data = conv_bias

    def forward(self, x):
        return self.conv(x)


class Reverse_Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels, out_channels * 4, ksize, stride, act=act)

    def forward(self, x):
        x1 = x.repeat(1, 1, 2, 2)
        x = self.conv(x)
        patch_top_left = x[:, ::4, ...]
        patch_bot_left = x[:, 1::4, ...]
        patch_top_right = x[:, 2::4, ...]
        patch_bot_right = x[:, 3::4, ...]
        # patch_top_left  = x[...,  ::2,  ::2]
        # patch_bot_left  = x[..., 1::2,  ::2]
        # patch_top_right = x[...,  ::2, 1::2]
        # patch_bot_right = x[..., 1::2, 1::2]
        # x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        x1[..., ::2, ::2] = patch_top_left
        x1[..., 1::2, ::2] = patch_bot_left
        x1[..., ::2, 1::2] = patch_top_right
        x1[..., 1::2, 1::2] = patch_bot_right
        return x1


class Non_local_Block(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.conv_out = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # g_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # theta_x: [N, HxW, C]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        # phi_x: [N,C,HxW]
        pairwise_weight = self.dot_product(theta_x, phi_x)
        # pairwise_weight: [N, HxW, HxW]
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)

        return output


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class Patch_Conv(nn.Module):
    def __init__(self, in_channel=256, out_channel=512, channel_scale=0.5,
                 patch_scale=2, stride=2, act="silu", channel_cat='linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)
        self.feat_patchconv_lt = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_lb = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_rt = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_rb = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)

        self.feat_patchconv_r = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_l = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_t = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_b = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(2 * self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(2 * self.middle_channel), out_channel, 1, 1, act=act)

    def forward(self, x):
        feat_patch_lt = x[:, :, :int(x.shape[2] / 2), :int(x.shape[3] / 2)]
        feat_patch_lb = x[:, :, int(x.shape[2] / 2):, :int(x.shape[3] / 2)]
        feat_patch_rt = x[:, :, :int(x.shape[2] / 2), int(x.shape[3] / 2):]
        feat_patch_rb = x[:, :, int(x.shape[2] / 2):, int(x.shape[3] / 2):]

        feat_patch_lt = self.feat_patchconv_lt(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb(feat_patch_rb)

        feat_patch_l = torch.cat((feat_patch_lt, feat_patch_lb), dim=2)
        feat_patch_r = torch.cat((feat_patch_rt, feat_patch_rb), dim=2)
        feat_patch_t = torch.cat((feat_patch_lt, feat_patch_rt), dim=3)
        feat_patch_b = torch.cat((feat_patch_lb, feat_patch_rb), dim=3)

        feat_patch_l = self.feat_patchconv_l(feat_patch_l)
        feat_patch_r = self.feat_patchconv_r(feat_patch_r)
        feat_patch_t = self.feat_patchconv_t(feat_patch_t)
        feat_patch_b = self.feat_patchconv_b(feat_patch_b)

        feat_patch_lr = torch.cat((feat_patch_l, feat_patch_r), dim=3)
        feat_patch_tb = torch.cat((feat_patch_t, feat_patch_b), dim=2)
        feat_patch = torch.cat((feat_patch_lr, feat_patch_tb), dim=1)
        feat_patch = self.channel_conv(feat_patch)

        return feat_patch


class Patch_Conv_NonLocal(nn.Module):
    def __init__(self, in_channel=256, out_channel=512, channel_scale=0.5,
                 patch_scale=2, stride=2, act="silu", channel_cat='linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)
        self.feat_patchconv_lt = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_lb = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_rt = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)
        self.feat_patchconv_rb = BaseConv(in_channel, self.middle_channel, 3, stride=stride, act=act)

        self.feat_patchconv_lt_nonlocal = Non_local_Block(in_channels=self.middle_channel,
                                                          inter_channels=self.middle_channel)
        self.feat_patchconv_lb_nonlocal = Non_local_Block(in_channels=self.middle_channel,
                                                          inter_channels=self.middle_channel)
        self.feat_patchconv_rt_nonlocal = Non_local_Block(in_channels=self.middle_channel,
                                                          inter_channels=self.middle_channel)
        self.feat_patchconv_rb_nonlocal = Non_local_Block(in_channels=self.middle_channel,
                                                          inter_channels=self.middle_channel)

        self.feat_patchconv_r = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_l = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_t = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        self.feat_patchconv_b = BaseConv(self.middle_channel, self.middle_channel, 3, stride=1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(2 * self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(2 * self.middle_channel), out_channel, 1, 1, act=act)

    def forward(self, x):
        feat_patch_lt = x[:, :, :int(x.shape[2] / 2), :int(x.shape[3] / 2)]
        feat_patch_lb = x[:, :, int(x.shape[2] / 2):, :int(x.shape[3] / 2)]
        feat_patch_rt = x[:, :, :int(x.shape[2] / 2), int(x.shape[3] / 2):]
        feat_patch_rb = x[:, :, int(x.shape[2] / 2):, int(x.shape[3] / 2):]

        feat_patch_lt = self.feat_patchconv_lt(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb(feat_patch_rb)

        feat_patch_lt = self.feat_patchconv_lt_nonlocal(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb_nonlocal(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt_nonlocal(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb_nonlocal(feat_patch_rb)

        feat_patch_l = torch.cat((feat_patch_lt, feat_patch_lb), dim=2)
        feat_patch_r = torch.cat((feat_patch_rt, feat_patch_rb), dim=2)
        feat_patch_t = torch.cat((feat_patch_lt, feat_patch_rt), dim=3)
        feat_patch_b = torch.cat((feat_patch_lb, feat_patch_rb), dim=3)

        feat_patch_l = self.feat_patchconv_l(feat_patch_l)
        feat_patch_r = self.feat_patchconv_r(feat_patch_r)
        feat_patch_t = self.feat_patchconv_t(feat_patch_t)
        feat_patch_b = self.feat_patchconv_b(feat_patch_b)

        feat_patch_lr = torch.cat((feat_patch_l, feat_patch_r), dim=3)
        feat_patch_tb = torch.cat((feat_patch_t, feat_patch_b), dim=2)
        feat_patch = torch.cat((feat_patch_lr, feat_patch_tb), dim=1)
        feat_patch = self.channel_conv(feat_patch)

        return feat_patch


if __name__ == '__main__':
    net = Non_local_Block(3, 3, 3)
    x = torch.rand([1, 3, 8, 8])
    # x1 = x.repeat(1,1,2,2)
    # print(x1.size())
    print('x=', x.size())
    y = net(x)
    print('y=', y.size())
    # print('p=', p.size())
