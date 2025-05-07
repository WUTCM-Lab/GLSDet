import torch
from torch import nn
from torch.nn import functional as F
from .darknet import BaseConv

class Non_local_Block(nn.Module):
    def __init__(self, in_channels, inter_channels = None):
        super().__init__()
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.g   = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
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
        n,_,h,w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)
        # g_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # theta_x: [N, HxW, C]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        # phi_x: [N,C,HxW]
        pairwise_weight = self.dot_product(theta_x, phi_x)
        # pairwise_weight: [N, HxW, HxW]
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0,2,1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)

        return output

class Patch_Conv_NonLocal(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, channel_scale = 0.5,
                 patch_scale = 2, stride = 2, act="silu", channel_cat = 'linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)
        self.feat_patchconv_lt = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_lb = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_rt = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_rb = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)

        self.feat_patchconv_lt_nonlocal = Non_local_Block(in_channels = self.middle_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_lb_nonlocal = Non_local_Block(in_channels = self.middle_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rt_nonlocal = Non_local_Block(in_channels = self.middle_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rb_nonlocal = Non_local_Block(in_channels = self.middle_channel, inter_channels=self.middle_channel)

        self.feat_patchconv_r = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_l = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_t = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_b = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(2 * self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(2 * self.middle_channel), out_channel, 1, 1, act=act)

    def forward(self, x):
        feat_patch_lt = x[:, :, :int(x.shape[2]/2), :int(x.shape[3]/2)]
        feat_patch_lb = x[:, :, int(x.shape[2]/2):, :int(x.shape[3]/2)]
        feat_patch_rt = x[:, :, :int(x.shape[2]/2), int(x.shape[3]/2):]
        feat_patch_rb = x[:, :, int(x.shape[2]/2):, int(x.shape[3]/2):]

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

class Patch_Conv_NonLocal_adapt(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, channel_scale = 1,
                 patch_scale = 2, stride = 2, act="silu", channel_cat = 'linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)

        self.attention_map = SpatialAttention()

        self.feat_patchconv_lt = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_lb = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_rt = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)
        self.feat_patchconv_rb = BaseConv(in_channel, self.middle_channel, 3, stride = stride, act=act)

        self.feat_patchconv_lt_nonlocal = Non_local_Block(in_channels = self.middle_channel)
        self.feat_patchconv_lb_nonlocal = Non_local_Block(in_channels = self.middle_channel)
        self.feat_patchconv_rt_nonlocal = Non_local_Block(in_channels = self.middle_channel)
        self.feat_patchconv_rb_nonlocal = Non_local_Block(in_channels = self.middle_channel)

        self.feat_patchconv_r = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_l = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_t = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_b = BaseConv(self.middle_channel, self.middle_channel, 3, stride = 1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(self.middle_channel), out_channel, 3, 1, act=act)

    def get_centroid(self, x):
        x_2 = x.sum(2)
        x_3 = x.sum(3)
        d = 0
        for i in range(x.shape[3]):
            d = x_2[:, :,i] + d
            if d.sum() > 0.5 * x.sum():
                break
        i = i // 2 * 2
        i = 4 if i < 4 else i
        i = x.shape[3]-4 if i > x.shape[3]-4 else i
        centroid_y = i
        d = 0
        for i in range(x.shape[2]):
            d = x_3[:, :, i] + d
            if d.sum() > 0.5 * x.sum():
                break
        i = i // 2 * 2
        i = 4 if i < 4 else i
        i = x.shape[2]-4 if i > x.shape[2]-4 else i
        centroid_x = i

        return centroid_x, centroid_y

    def forward(self, x):
        attention_map = self.attention_map(x)
        max_value = attention_map.max()
        min_value = attention_map.min()
        threshold_value = min_value + 0.75 * (max_value - min_value)
        attention_map[attention_map < threshold_value] = 0
        centroid_x, centroid_y = self.get_centroid(attention_map)
        attention_patch_l = attention_map[:, :, :centroid_x, :]
        attention_patch_r = attention_map[:, :, centroid_x:, :]
        centroid_x_l, centroid_y_l = self.get_centroid(attention_patch_l)
        centroid_x_r, centroid_y_r = self.get_centroid(attention_patch_r)
        feat_patch_lt = x[:, :, :centroid_x, :centroid_y_l]
        feat_patch_lb = x[:, :, centroid_x:, :centroid_y_r]
        feat_patch_rt = x[:, :, :centroid_x, centroid_y_l:]
        feat_patch_rb = x[:, :, centroid_x:, centroid_y_r:]

        feat_patch_lt = self.feat_patchconv_lt(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb(feat_patch_rb)

        feat_patch_lt = self.feat_patchconv_lt_nonlocal(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb_nonlocal(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt_nonlocal(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb_nonlocal(feat_patch_rb)

        feat_patch_t = torch.cat((feat_patch_lt, feat_patch_rt), dim=3)
        feat_patch_b = torch.cat((feat_patch_lb, feat_patch_rb), dim=3)


        feat_patch_t = self.feat_patchconv_t(feat_patch_t)
        feat_patch_b = self.feat_patchconv_b(feat_patch_b)


        feat_patch = torch.cat((feat_patch_t, feat_patch_b), dim=2)

        feat_patch = self.channel_conv(feat_patch)

        return feat_patch

class Patch_Conv_NonLocal_new(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, channel_scale = 0.5,
                 patch_scale = 2, act="silu", channel_cat = 'non_linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)
        self.feat_patchconv_lt_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_lb_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rt_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rb_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)

        # self.feat_patchconv_r = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        # self.feat_patchconv_l = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        # self.feat_patchconv_t = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        # self.feat_patchconv_b = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(self.middle_channel), out_channel, 3, 1, act=act)

    def forward(self, x):
        feat_patch_lt = x[:, :, :int(x.shape[2]/2), :int(x.shape[3]/2)]
        feat_patch_lb = x[:, :, int(x.shape[2]/2):, :int(x.shape[3]/2)]
        feat_patch_rt = x[:, :, :int(x.shape[2]/2), int(x.shape[3]/2):]
        feat_patch_rb = x[:, :, int(x.shape[2]/2):, int(x.shape[3]/2):]

        feat_patch_lt = self.feat_patchconv_lt_nonlocal(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb_nonlocal(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt_nonlocal(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb_nonlocal(feat_patch_rb)

        feat_patch_t = torch.cat((feat_patch_lt, feat_patch_rt), dim=3)
        feat_patch_b = torch.cat((feat_patch_lb, feat_patch_rb), dim=3)


        # feat_patch_t = self.feat_patchconv_t(feat_patch_t)
        # feat_patch_b = self.feat_patchconv_b(feat_patch_b)

        feat_patch = torch.cat((feat_patch_t, feat_patch_b), dim=2)

        feat_patch = self.channel_conv(feat_patch)
        return feat_patch

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = Patch_Conv_NonLocal_new(d_model, d_model, channel_scale=1,
            patch_scale=2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # print(x.shape)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Patch_Conv_NonLocal_adapt_new(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, channel_scale = 0.5,
                 patch_scale = 2, act="silu", channel_cat = 'non_linear'):
        super().__init__()
        # self.input_shape = inputshape
        self.channel_scale = channel_scale
        self.patch_scale = patch_scale
        # self.feat_patchsize = int(self.input_shape / 8 / self.patch_scale)
        self.middle_channel = int(self.channel_scale * in_channel)

        self.attention_map = SpatialAttention()

        self.feat_patchconv_lt_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_lb_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rt_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)
        self.feat_patchconv_rb_nonlocal = Non_local_Block(in_channels = in_channel, inter_channels=self.middle_channel)

        self.feat_patchconv_r = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_l = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_t = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        self.feat_patchconv_b = BaseConv(in_channel, self.middle_channel, 3, stride = 1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(self.middle_channel), out_channel, 3, 1, act=act)

    def get_centroid(self, x):
        with torch.no_grad():
            x_2 = x.sum(2)
            x_3 = x.sum(3)
            d = 0
            for i in range(x.shape[3]):
                d = x_2[:, :,i] + d
                if d.sum() > 0.5 * x.sum():
                    break
            i = i // 2 * 2
            i = 4 if i < 4 else i
            i = x.shape[3]-4 if i > x.shape[3]-4 else i
            centroid_y = i
            d = 0
            for i in range(x.shape[2]):
                d = x_3[:, :, i] + d
                if d.sum() > 0.5 * x.sum():
                    break
            i = i // 2 * 2
            i = 4 if i < 4 else i
            i = x.shape[2]-4 if i > x.shape[2]-4 else i
            centroid_x = i

        return centroid_x, centroid_y

    def forward(self, x):
        attention_map = self.attention_map(x)
        max_value = attention_map.max()
        min_value = attention_map.min()
        threshold_value = min_value + 0.75 * (max_value - min_value)
        attention_map[attention_map < threshold_value] = 0
        centroid_x, centroid_y = self.get_centroid(attention_map)
        attention_patch_l = attention_map[:, :, :centroid_x, :]
        attention_patch_r = attention_map[:, :, centroid_x:, :]
        centroid_x_l, centroid_y_l = self.get_centroid(attention_patch_l)
        centroid_x_r, centroid_y_r = self.get_centroid(attention_patch_r)
        feat_patch_lt = x[:, :, :centroid_x, :centroid_y_l]
        feat_patch_lb = x[:, :, centroid_x:, :centroid_y_r]
        feat_patch_rt = x[:, :, :centroid_x, centroid_y_l:]
        feat_patch_rb = x[:, :, centroid_x:, centroid_y_r:]

        feat_patch_lt = self.feat_patchconv_lt_nonlocal(feat_patch_lt)
        feat_patch_lb = self.feat_patchconv_lb_nonlocal(feat_patch_lb)
        feat_patch_rt = self.feat_patchconv_rt_nonlocal(feat_patch_rt)
        feat_patch_rb = self.feat_patchconv_rb_nonlocal(feat_patch_rb)

        feat_patch_t = torch.cat((feat_patch_lt, feat_patch_rt), dim=3)
        feat_patch_b = torch.cat((feat_patch_lb, feat_patch_rb), dim=3)


        feat_patch_t = self.feat_patchconv_t(feat_patch_t)
        feat_patch_b = self.feat_patchconv_b(feat_patch_b)


        feat_patch = torch.cat((feat_patch_t, feat_patch_b), dim=2)

        feat_patch = self.channel_conv(feat_patch)
        attention_map = self.attention_map(x)
        feat_patch = attention_map * feat_patch
        return feat_patch

class Patch_Conv_NonLocal_44(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, channel_scale = 0.5,
                 patch_scale = 2, stride = 2, act="silu", channel_cat = 'linear'):
        super().__init__()
        self.channel_scale = channel_scale
        # self.input_shape = inputshape
        self.middle_channel = int(self.channel_scale * in_channel)
        self.patchconv_lt_nonlocal = Patch_Conv_NonLocal(
            in_channel = in_channel,
            out_channel = out_channel,
            patch_scale = patch_scale
        )
        self.patchconv_lb_nonlocal = Patch_Conv_NonLocal(
            in_channel = in_channel,
            out_channel = out_channel,
            patch_scale = patch_scale
        )
        self.patchconv_rt_nonlocal = Patch_Conv_NonLocal(
            in_channel = in_channel,
            out_channel = out_channel,
            patch_scale = patch_scale
        )
        self.patchconv_rb_nonlocal = Patch_Conv_NonLocal(
            in_channel = in_channel,
            out_channel = out_channel,
            patch_scale = patch_scale
        )
        self.feat_patchconv_r = BaseConv(int(4 * self.middle_channel), self.middle_channel, 1, stride = 1, act=act)
        self.feat_patchconv_l = BaseConv(int(4 * self.middle_channel), self.middle_channel, 1, stride = 1, act=act)
        self.feat_patchconv_t = BaseConv(int(4 * self.middle_channel), self.middle_channel, 1, stride = 1, act=act)
        self.feat_patchconv_b = BaseConv(int(4 * self.middle_channel), self.middle_channel, 1, stride = 1, act=act)
        if channel_cat == 'linear':
            self.channel_conv = nn.Conv2d(int(2 * self.middle_channel), out_channel, 1, 1)
        else:
            self.channel_conv = BaseConv(int(2 * self.middle_channel), out_channel, 1, 1, act=act)

    def forward(self, x):
        feat_patch_lt = x[:, :, :int(x.shape[2]/2), :int(x.shape[3]/2)]
        feat_patch_lb = x[:, :, int(x.shape[2]/2):, :int(x.shape[3]/2)]
        feat_patch_rt = x[:, :, :int(x.shape[2]/2), int(x.shape[3]/2):]
        feat_patch_rb = x[:, :, int(x.shape[2]/2):, int(x.shape[3]/2):]

        feat_patch_lt = self.patchconv_lt_nonlocal(feat_patch_lt)
        feat_patch_lb = self.patchconv_lb_nonlocal(feat_patch_lb)
        feat_patch_rt = self.patchconv_rt_nonlocal(feat_patch_rt)
        feat_patch_rb = self.patchconv_rb_nonlocal(feat_patch_rb)

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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



if __name__ == '__main__':
    net = Non_local_Block(3, 3, 3)
    x = torch.rand([1,3,8,8])
    # x1 = x.repeat(1,1,2,2)
    # print(x1.size())
    print('x=',x.size())
    y = net(x)
    print('y=',y.size())
    # print('p=', p.size())