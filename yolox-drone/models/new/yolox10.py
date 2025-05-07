import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decouple.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

from models.new.Non_local_family import Patch_Conv_NonLocal_new

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        self.csp_feat0 = CSPLayer(
            int(0.5 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * 0.75),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.up_convs = nn.ModuleList()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.up4 = nn.Upsample(scale_factor=4, mode="nearest")
        # self.ups = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))

            self.up_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=2, act=act),
            ]))


            if i == 2:
                self.cls_convs.append(nn.Sequential(*[
                    Conv(in_channels=int(256 * width * 2), out_channels=int(256 * width * 2), ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width * 2), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                ]))
            else:
                self.cls_convs.append(nn.Sequential(*[
                    Conv(in_channels=int(256 * width * 3), out_channels=int(256 * width * 3), ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width * 3), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                ]))

            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width ), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        inputs_processed = []
        feat0_processed = None
        for k, x in enumerate(inputs):
            if k == 0:
                feat0_processed = self.csp_feat0(x)
            else:
                print(x.shape)
                y = self.stems[k-1](x)
                inputs_processed.append(y)
                print(y.shape)


        for k, x in enumerate(inputs_processed):
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            if k == 0:
                cls_feat = torch.cat([x, self.up_convs[k](feat0_processed), self.up(inputs_processed[k+1])], 1)
            elif k == 1:
                cls_feat = torch.cat([x, self.up_convs[k](inputs_processed[k-1]), self.up(inputs_processed[k+1])], 1)
            else:
                cls_feat = torch.cat(
                    [x, self.up_convs[k](inputs_processed[k - 1])], 1)
            # print(cls_feat.shape)
            cls_feat = self.cls_convs[k](cls_feat)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            # temp = []
            # if k == 0:
            #     #nl0_down = F.adaptive_max_pool2d(feat0_processed, (80, 80))
            #     # l0_up = self.up(inputs_processed[k+1])
            #     # temp.append(l0_up)
            #     temp.append(self.ups[k](feat0_processed))
            #     # corner_feature = torch.stack((x, l0_up, l0_down), dim=1)
            # elif k == 1:
            #     # l1_up = self.up(inputs_processed[k+1])
            #     # #nl2_down = F.adaptive_max_pool2d(inputs_processed[k-1], (40, 40))
            #     # temp.append(l1_up)
            #     temp.append(self.ups[k](inputs_processed[k - 1]))
            #     # corner_feature = torch.stack((x, l1_up, l2_down), dim=1)
            # else:
            #     # l3_down = F.adaptive_max_pool2d(inputs_processed[k - 1], (20, 20))
            #     temp.append(self.ups[k](inputs_processed[k-1]))
            #     #ncorner_feature = torch.stack((x, l3_down), dim=1)
            #
            # # print(corner_feature.size())
            #
            # # print(x.shape)
            # for temp_feature in temp:
            #     x += temp_feature
            #     # print(temp_feature.shape)
            #
            # # print("\n")
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.Patch_conv_feat1 = Patch_Conv_NonLocal_new(
            in_channel=int(in_channels[0] * width),
            out_channel=int(in_channels[0] * width),
            channel_scale=1,
            patch_scale=2
        )
        #
        self.Patch_conv_feat2 = Patch_Conv_NonLocal_new(
            in_channel=int(in_channels[1] * width),
            out_channel=int(in_channels[1] * width),
            channel_scale=1,
            patch_scale=2
        )

        self.Patch_conv_feat3 = Patch_Conv_NonLocal_new(
            in_channel=int(in_channels[2] * width),
            out_channel=int(in_channels[2] * width),
            channel_scale=1,
            patch_scale=2
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat0, feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        # print(feat1.shape)
        feat1 = feat1 + self.Patch_conv_feat1(feat1)
        # print(feat2.shape)
        feat2 = feat2 + self.Patch_conv_feat2(feat2)
        # print(feat3.shape)
        feat3 = feat3 + self.Patch_conv_feat3(feat3)

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat3)
        # -------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.upsample(P5)
        # -------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        P3_out = self.C3_p3(P4_upsample)

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        P3_downsample = self.bu_conv2(P3_out)
        # -------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P4_out = self.C3_n3(P3_downsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        P4_downsample = self.bu_conv1(P4_out)
        # -------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        # -------------------------------------------#
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        P5_out = self.C3_n4(P4_downsample)

        return (feat0, P3_out, P4_out, P5_out)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs
