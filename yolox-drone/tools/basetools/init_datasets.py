import os
import shutil

SRC_IMAGES = '/Users/sunhaokai/Downloads/GF6_2m_20220314/test'
SRC_ANNOTATIONS = '/Users/sunhaokai/Downloads/GF6_2m_20220314/testxml'
MODE = '.jpg'

DST_PATH = 'hn1h_2_test'
SECOND_DIR = 'VOC2007'
ANNOTATIONS_DIR = 'Annotations'
TXT_DIR = 'ImageSets'
IMAGES_DIR = 'JPEGImages'


if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)
    sec_dir = os.path.join(DST_PATH, SECOND_DIR)
    os.makedirs(sec_dir)
    ann_dir = os.path.join(sec_dir, ANNOTATIONS_DIR)
    txt_dir = os.path.join(sec_dir, TXT_DIR)
    images_dir = os.path.join(sec_dir, IMAGES_DIR)
    third_dirs = [ann_dir, txt_dir, images_dir]
    for i in third_dirs:
        os.makedirs(i)
    os.makedirs(os.path.join(txt_dir, 'Main'))

sec_dir = os.path.join(DST_PATH, SECOND_DIR)
num = 0
for file in os.listdir(SRC_ANNOTATIONS):
    src_ann_file = os.path.join(SRC_ANNOTATIONS, file)
    dst_ann_file = os.path.join(os.path.join(sec_dir, ANNOTATIONS_DIR), file)
    shutil.copy(src_ann_file, dst_ann_file)

    src_image_file = os.path.join(SRC_IMAGES, file.replace('.xml', MODE))
    if not os.path.exists(src_image_file):
        print(file + ' is not existed!')
        continue
    dst_image_file = os.path.join(os.path.join(sec_dir, IMAGES_DIR), file.replace('.xml', MODE))
    shutil.copy(src_image_file, dst_image_file)
    num += 1
    print(file + ' is processed.')

print('Total number: ' + str(num))





