import argparse
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import warnings
import cv2
import mmcv
import torch
import math
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import numpy as np
from mmdet.core import UnifiedForegroundPacking
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
import json

CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 255),
    (128, 255, 128),
    (255, 128, 128),
    (255, 255, 255),

]


def compute_iof(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / min(area1, area2)


def compute_iou(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / (area1 + area2 - inter)



# modify test
class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results, bbox=None, img_data=None):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None

        if img_data is None:
            img = mmcv.imread(results['img'])
        else:
            img = img_data
        if bbox:
            x1, x2, y1, y2, _ = bbox
            img = img[x1:x2, y1:y2, :]
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def my_inference_detector(model, data):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result[0]


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



def display_merge_result(results, img, img_name, w, h):
    w = math.ceil(w)
    h = math.ceil(h)
    img_data = cv2.imread(img)
    new_img = np.zeros((h, w, 3))
    for result in results:
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))
    return new_img


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-coarse_detector_config', help='')
    parser.add_argument('-coarse_detector_config_ckpt', help='')
    parser.add_argument('-mp_det_config', help='')
    parser.add_argument('-mp_det_config_ckpt', help='')
    parser.add_argument('-dataset_anno', help='')
    parser.add_argument('-dataset_root', help='')
    args = parser.parse_args()
    return args


def main():
    device = 'cuda'
    args = parse_args()
    coarse_detecor_config = args.coarse_detector_config
    coarse_detecor_config_ckpt = args.coarse_detector_config_ckpt
    mp_det_config = args.mp_det_config
    mp_det_config_ckpt = args.mp_det_config_ckpt
    dataset_anno = args.dataset_anno
    dataset_root = args.dataset_root

    coarse_detecor_config = './configs/UFPMP-Det/coarse_det.py'
    coarse_detecor_config_ckpt = '/home/server2/shk/rawdet.pth'
    mp_det_config = './configs/UFPMP-Det/mp_det_res50.py'
    mp_det_config_ckpt = '/home/server2/shk/det.pth'
    dataset_anno = '/home/server2/shk/VisDrone/VisDrone_Dataset_COCO_Format/annotations/instances_UAVval.json'
    dataset_root = '/home/server2/shk/VisDrone/VisDrone_Dataset_COCO_Format/images/instances_UAVval/images'


    coarse_detecor = init_detector(coarse_detecor_config, coarse_detecor_config_ckpt, device=device)
    mp_det = init_detector(mp_det_config, mp_det_config_ckpt, device=device)

    with open(dataset_anno) as f:
        json_info = json.load(f)
    annotation_set = {}
    for annotation in json_info['annotations']:
        image_id = annotation['image_id']
        if not image_id in annotation_set.keys():
            annotation_set[image_id] = []
        annotation_set[image_id].append(annotation)

    coco = COCO(dataset_anno)  # 导入验证集
    size = len(list(coco.imgs.keys()))
    results = []
    times = []
    # shape_set = set()
    cnt = 1
    rm_cnt = 1e-6
    tp_cnt = 1e-6
    sum_rm = 0
    sum_tp = 0
    # print(size)
    for key in range(size):
        print(cnt, size, end='\r')
        cnt += 1
        # if cnt > 10:
        #     continue
        image_id = key
        width = coco.imgs[key]['width']
        height = coco.imgs[key]['height']
        img_name = coco.imgs[key]['file_name']
        img = os.path.join(dataset_root, img_name)
        data = dict(img=img)
        cur_annotation = annotation_set[key]
        first_results = my_inference_detector(coarse_detecor, LoadImage()(data))
        result = np.concatenate(first_results)
        rec, w, h = UnifiedForegroundPacking(result[:, :4], 1.5, input_shape=[width, height])
        next_image = display_merge_result(rec, img, img_name, w, h)

        # print(next_image.shape)

        # ignore_list = gen_ignore_list(img_name)
        time2 = time.time()
        second_results = my_inference_detector(mp_det, LoadImage()(data, img_data=next_image))

        # print(second_results)
        time3 = time.time()
        times.append(time3-time2)
        finale_results = []
        for first_result in first_results:
            finale_results.append(first_result)
        new_second_result = []
        for i in range(10):
            new_second_result.append([])

        for chips in rec:
            o_x1, o_y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in chips]
            chip_bbox = [n_x, n_y, n_x + w * scale_factor, n_y + h * scale_factor]
            for idx, _results in enumerate(second_results):
                for _result in _results:
                    x1, y1, x2, y2, score = _result
                    # print(x1,y1,x2,y2, score)
                    t_bbox = [x1, y1, x2, y2]
                    if compute_iof(t_bbox, chip_bbox) > 0.9:
                        new_w = (x2 - x1) / scale_factor
                        new_h = (y2 - y1) / scale_factor
                        new_x = (x1 - n_x) / scale_factor + o_x1
                        new_y = (y1 - n_y) / scale_factor + o_y1
                        new_bbox = [new_x, new_y, new_x + new_w, new_y + new_h, score]
                        new_second_result[idx].append(new_bbox)

        finale_results = new_second_result
        for idx in range(len(finale_results)):
            finale_results[idx] = np.array(finale_results[idx])
        # display_result(finale_results, img, img_name)
        for idx, result in enumerate(finale_results):
            result = np.array(result)
            if result.shape[0] == 0:
                continue
            keep = py_cpu_nms(result,0.6)
            for bbox in result[keep]:
                rm_cnt += 1
                x1, y1, x2, y2, score = bbox
                sum_rm += score
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                pred_bbox = [x1,y1,x2,y2]
                image_result = {
                    'image_id': image_id,
                    'category_id': idx,
                    'score': float(score),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                }
                results.append(image_result)
        # append detection to results
    # write output
    print(sum(times)/len(times))
    json.dump(results, open('{}_bbox_result_tmp.json'.format('UAV'), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = coco
    coco_pred = coco_true.loadRes('{}_bbox_result_tmp.json'.format('UAV'))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    # coco_eval.params.imgIds = image_ids
    coco_eval.params.maxDets = [10, 100 , 500]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
