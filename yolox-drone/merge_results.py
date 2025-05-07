import os
import time

import torch
from torchvision.ops import boxes
import numpy as np

CLASS_PATH = '/home/server2/shk/yolox_drone/model_data/visdrone10.txt'
CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = {c.strip():i  for i, c in enumerate(class_names)}
    return class_names, len(class_names)
class_book, class_num = get_classes(CLASS_PATH)


def get_annotations(txt_path, prob=True):
    bbox = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            info = line[:-1].split(' ')
            tmp = []
            if prob:
                for i in range(2, 6):
                    tmp.append(float(info[i]))
                tmp.append(float(info[1]))
            else:
                for i in range(1, 5):
                    tmp.append(float(info[i]))
                tmp.append(1)
            id = class_book[info[0]]
            tmp.append(id)
            bbox.append(tmp)
    return bbox

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep

def batched_soft_nms(boxes,scores,idxs) :
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        # curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        curr_keep_indices = py_cpu_softnms(boxes[curr_indices].numpy(), scores[curr_indices].numpy(), Nt=0.3, sigma=0.5, thresh=0.0001, method=2)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


if __name__ == "__main__":
    dir_src = '/home/server2/shk/yolox_drone/0224uav_ufp_results'
    dir_src2 = '/home/server2/shk/yolox_drone/new_results/uav_ftt2_30/detection-results'
    dir_out = '/home/server2/shk/yolox_drone/new_results/final_uav_mix'
    dirs = [dir_src, dir_src2]
    print(dirs)
    image_list = os.listdir(dirs[0])
    cnt = 1
    for name in image_list:
        print(cnt)
        cnt += 1
        cur = []
        for tmp_dir in dirs:
            txt_path = os.path.join(tmp_dir, name)
            result = get_annotations(txt_path)
            result = torch.Tensor(result)
            cur.append(result)
            #print(result.shape)
        cur = torch.cat(cur, dim=0)
        detections = cur
        positive_indexes = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4],
            detections[:, 5],
            0.65,
        )
        # positive_indexes = batched_soft_nms(
        #     detections[:, :4],
        #     detections[:, 4],
        #     detections[:, 5],
        # )
        detections = detections[positive_indexes]
        #print(detections.shape)
        out_path = os.path.join(dir_out, name)
        f = open(out_path, "w")
        for i in range(detections.shape[0]):
            detections_t = detections[i]
            predicted_class = CLASSES[int(detections_t[5])]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, float(detections_t[4]), int(detections_t[0]), int(detections_t[1]), int(detections_t[2]), int(detections_t[3])))
            # predicted_class, score[:6], str(left), str(top), str(right), str(bottom)))
        f.close()
        # exit()









