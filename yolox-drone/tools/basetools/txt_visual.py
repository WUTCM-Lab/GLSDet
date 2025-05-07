import os
import cv2
import colorsys

import numpy as np
from PIL import ImageDraw, ImageFont, Image

from models.core.utils import cvtColor


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = {c.strip():i  for i, c in enumerate(class_names)}
    return class_names, len(class_names)


classes, num_classes = get_classes('model_data/visdrone10.txt')
print(classes)
hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
new_colors = []
for i in range(len(colors)):
    if i == 4:
        new_colors.append((56,87,35))
    else:
        new_colors.append(colors[i])
colors = new_colors

def get_annotations(txt_path, prob=False):
    bbox = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            info = line[:-1].split(' ')
            tmp = []
            tmp.append(info[0])
            if prob:
                for i in range(2, 6):
                    tmp.append(int((info[i])))
                tmp.append(info[1])
            else:
                for i in range(1, 5):
                    tmp.append(int((info[i])))
                tmp.append(int(1))
            bbox.append(tmp)
    return bbox


def save_viz_image(image_path, txt_path, save_path):
    bbox = get_annotations(txt_path)
    image = Image.open(image_path)
    image = cvtColor(image)
    draw = ImageDraw.Draw(image)
    thickness = 2
    for info in bbox:
        box_class = classes[info[0]]
        for i in range(thickness):
            draw.rectangle([info[1]+i, info[2]+i, info[3]-i, info[4]-i], outline=colors[box_class])
        # cv2.putText(image, info[0] + ':' + info[5], (info[1], info[2]), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
    del draw
    image.save(save_path)

def save_viz_image_example(image_path, txt_path, save_path, k):
    bbox = get_annotations(txt_path)
    image = Image.open(image_path)
    image = cvtColor(image)
    draw = ImageDraw.Draw(image)
    thickness = 2
    for info in bbox:
        box_class = classes[info[0]]
        for i in range(thickness):
            draw.rectangle([info[1]+i, info[2]+i, info[3]-i, info[4]-i], outline=colors[k])
        # cv2.putText(image, info[0] + ':' + info[5], (info[1], info[2]), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
    del draw
    image.save(save_path)



if __name__ == '__main__':
    image_dir ='VOCdevkit_val/VOC2007/JPEGImages'
    txt_dir = 'results/tmp62307/ground-truth'

    save_dir = 'gt1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_list = os.listdir(image_dir)
    for i in image_list:
        image_path = os.path.join(image_dir, i)
        txt_path = os.path.join(txt_dir, i.replace('.jpg', '.txt'))
        save_path = os.path.join(save_dir, i)
        save_viz_image(image_path, txt_path, save_path)

    save_path = os.path.join(save_dir, i)
    # image_path = 'VOCdevkit_val/VOC2007/JPEGImages/0000001_02999_d_0000005.jpg'
    #
    # txt_path = 'results/tmp23805/ground-truth/0000001_02999_d_0000005.txt'
    # save_path = 'example_'
    # for i in range(num_classes):
    #     path = str(i)+save_path + list(classes.keys())[i] + '.jpg'
    #     save_viz_image_example(image_path, txt_path, path, i)


