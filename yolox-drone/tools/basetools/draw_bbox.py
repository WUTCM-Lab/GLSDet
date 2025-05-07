import os
from xml.etree import ElementTree as ET
import cv2
import sys


class DrawBBox(object):
    """
    draw bound boxes on images and save images
    provide two kinds of formats of src: CoCo(.json) or VOC(dir/*.xml)
    """
    def __init__(self, src_path, img_dir, dst_dir, mode='CoCo', image_mode='.jpg'):
        self.mode = mode
        self.src_path = src_path
        self.img_dir = img_dir
        self.dst_dir = dst_dir
        self.image_mode = image_mode

        # check the paths
        if not os.path.exists(self.src_path):
            raise Exception(src_path + ' is not existed.')
        if not os.path.exists(self.img_dir):
            raise Exception(img_dir + ' is not existed.')
        if not os.path.exists(self.dst_dir):
            os.mkdir(dst_dir)

    def draw_by_voc(self):
        sum_files = len(os.listdir(self.src_path))
        for index, xml_file in enumerate(os.listdir(self.src_path)):
            xml_path = os.path.join(self.src_path, xml_file)
            src_img_path = os.path.join(self.img_dir, xml_file.replace('.xml', self.image_mode))
            if not os.path.exists(xml_path):
                print(xml_path + ' is not existed.')
                continue
            if not os.path.exists(src_img_path):
                print(src_img_path + ' is not existed.')
                continue

            sys.stdout.write('\rprocessed>>>>%d/%d' % (index+1, sum_files))
            sys.stdout.flush()

            xml_parser = ET.parse(xml_path)
            root = xml_parser.getroot()
            objects = root.findall('object')
            image = cv2.imread(src_img_path)
            for obj in objects:
                bndbox = obj.find('bndbox')
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('ymax').text)
                name = obj.find('name').text
                area = (y2 - y1) * (x2 - x1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                #cv2.putText(image, str(area), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

            dst_image_path = os.path.join(self.dst_dir, xml_file.replace('.xml', self.image_mode))
            cv2.imwrite(dst_image_path, image)

    def __call__(self, *args, **kwargs):
        if self.mode == 'VOC':
            self.draw_by_voc()


if __name__ == '__main__':
    # how to use?
    draw_bbox = DrawBBox('/home/server2/shk/yolox_drone/images/outputs', '/home/server2/shk/yolox_drone/images/images', 'show', 'VOC')
    draw_bbox()