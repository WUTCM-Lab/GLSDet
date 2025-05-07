import os
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


if __name__ == "__main__":
    dir_src = '/home/server2/shk/yolox_drone/results/ftt768/detection-results'
    dir_src2 = '/home/server2/shk/yolox_drone/results/ftt640/detection-results'
    dir_out = '/home/server2/shk/yolox_drone/results/0000vis'
    dirs = [dir_src, dir_src2]

    image_list = os.listdir(dirs[0])
    cnt = 1
    for name in image_list:
        print(cnt)
        cnt+=1
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









