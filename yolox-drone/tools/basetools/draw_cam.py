import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
# from models.decouple.yolox_recls import YoloBody
from models.base.yolox import YoloBody


def hook_feature(module, input, output):  # hook注册, 响应图提取
    print("hook input", output.shape)
    features_blobs.append(input)


def returnCAM(feature_conv, size_upsample):
    # 生成CAM图: 输入是feature_conv和weight_softmax
    bz, nc, h, w = feature_conv[0].shape
    weights = F.adaptive_avg_pool2d(feature_conv[0], (1, 1))
    cam = weights * feature_conv[0]
    cam = torch.sum(cam, dim=1)
    cam = cam.reshape(h, w).detach().numpy()
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    result = cv2.resize(cam_img, size_upsample)

    return result


if __name__ == '__main__':

    size_upsample = (640, 640)
    stage_num = 3
    SAVE_DIR = 'src2'
    img_name = '/home/server2/shk/yolox_drone/VOCdevkit_val/VOC2007/JPEGImages/0000244_03000_d_0000007.jpg'
    model_path = '/home/server2/shk/yolox_drone/VOCdevkit_val/VOC2007/JPEGImages/0000244_03000_d_0000007.jpg'

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(size_upsample),
        transforms.ToTensor(),
        normalize])

    img_pil = Image.open(img_name)
    img = cv2.imread(img_name)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    net = YoloBody(10, 'm')
    net.load_state_dict(torch.load(model_path),
                        strict=True)
    net.eval()
    features_blobs = []
    for i in range(stage_num):
        getattr(net.head.cls_convs, str(i)).register_forward_hook(hook_feature)
    net(img_variable)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    for i in range(stage_num):
        CAMs = returnCAM(features_blobs[i], size_upsample)
        img = cv2.resize(img, size_upsample)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)

        heatmap_path = os.path.join(SAVE_DIR, 'heatmap_{}.jpg'.format(str(i)))
        cam_path = os.path.join(SAVE_DIR, 'cam_{}.jpg'.format(str(i)))
        cv2.imwrite(heatmap_path, heatmap)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite(cam_path, result)