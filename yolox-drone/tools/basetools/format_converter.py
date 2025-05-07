import os
import json
import glob
import xml.etree.ElementTree as ET
import sys

"""
The following shows templates of .xml file.
"""
HEAD = """\
<annotation>
    <folder>%s</folder>
    <filename>%s</filename>
    <path>%s</path>
    <source>
        <database>%s</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
OBJ = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

TAIL = '''\
</annotation>
'''


class FormatConverter(object):

    """
    provide two modes: transfer *.json to .xml (CoCo to Voc)
                       transfer dir_name/*.xml to .json (Voc to CoCo)

    mode: coco2voc, voc2coco

    """

    def __init__(self, src_path, dst_path, mode='coco2voc', keys=None,
                 image_mode='.jpg', init_bbox_id=1, classes=None, classes_path='classes.txt'):
        self.mode = mode
        self.src_path = src_path
        self.dst_path = dst_path
        self.keys = keys
        self.image_mode = image_mode
        self.image_channels = 3
        self.database = 'Unknown'
        self.init_bbox_id = init_bbox_id
        self.classes_path = classes_path

        '''
        for mode is voc2coco
        '''
        self.classes = classes

        # check the paths
        if not os.path.exists(self.src_path):
            raise Exception(src_path + ' is not existed.')

        # default
        if keys is None:
            self.keys = dict(image='images', annotation='annotations', category='categories')

        if mode == 'coco2voc':
            """
            templates:
            """
            self.head = HEAD
            self.obj = OBJ
            self.tail = TAIL
            if not os.path.exists(self.dst_path):
                os.mkdir(dst_path)

    def coco2voc(self):

        with open(self.src_path, encoding="utf-8") as load_f:
            json_dict = json.load(load_f)
        categories = json_dict[self.keys['category']]

        ff = open(self.classes_path, 'w')
        ff.write('(')
        index = 1
        for i in categories:
            ff.write('\''+i['name']+'\'')
            if index != len(categories):
                ff.write(',')
            if index % 5 == 0:
                ff.write('\n')
            index += 1
        ff.write(')')
        ff.close()

        annotations = json_dict[self.keys['annotation']]
        images = json_dict[self.keys['image']]

        annotation_index = 0
        annotation_num = len(annotations)

        for image in images:
            file_name = image['file_name'].replace('.jpg', '.xml')
            xml_path = os.path.join(self.dst_path, file_name)
            image_id = image['id']

            f_xml = open(xml_path, "w")

            sys.stdout.write('\r>> Converting file %d/%d' % (
                image_id + 1, len(images)))
            sys.stdout.flush()

            temp_head = self.head % (self.dst_path,
                                     file_name[:-4],
                                     xml_path,
                                     self.database,
                                     image['width'],
                                     image['height'],
                                     self.image_channels)
            f_xml.write(temp_head)

            cur_index = 0
            for temp_index in range(annotation_index, annotation_num):
                cur_index = temp_index
                if annotations[temp_index]['image_id'] != image_id:
                    break
                annotation = annotations[temp_index]
                for category in categories:
                    if category['id'] == annotation['category_id']:
                        name = category['name']
                        break

                '''
                bbox = [x,y,width,height]
                '''
                bbox = annotation['bbox']
                temp_obj = self.obj % (name, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                f_xml.write(temp_obj)

            annotation_index = cur_index

            f_xml.write(self.tail)

    def voc2coco(self):
        def get(root, name):
            return root.findall(name)

        def get_and_check(root, name, length):
            vars = root.findall(name)
            if len(vars) == 0:
                raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
            if length > 0 and len(vars) != length:
                raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
            if length == 1:
                vars = vars[0]
            return vars

        json_dict = {'images': [], 'type': 'instances',
                     'categories': [], 'annotations': []}
        categories = self.classes
        bbox_id = self.init_bbox_id
        xml_paths = glob.glob(os.path.join(self.src_path, '*.xml'))
        for image_id, xml_f in enumerate(xml_paths):

            sys.stdout.write('\r>> Converting file %d/%d' % (
                image_id + 1, len(xml_paths)))
            sys.stdout.flush()

            tree = ET.parse(xml_f)
            root = tree.getroot()
            filename = get_and_check(root, 'filename', 1).text
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {'file_name': filename + '.jpg', 'height': height,
                     'width': width, 'id': image_id + 1}
            json_dict['images'].append(image)

            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category not in categories:
                    new_id = max(categories.values()) + 1
                    categories[category] = new_id
                category_id = categories[category]
                bbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bbox, 'xmin', 1).text)
                ymin = int(get_and_check(bbox, 'ymin', 1).text)
                xmax = int(get_and_check(bbox, 'xmax', 1).text)
                ymax = int(get_and_check(bbox, 'ymax', 1).text)
                if xmax <= xmin or ymax <= ymin:
                    continue
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id + 1,
                       'bbox': [xmin, ymin, o_width, o_height], 'category_id': category_id,
                       'id': bbox_id, 'ignore': 0, 'segmentation': []}
                json_dict['annotations'].append(ann)
                bbox_id = bbox_id + 1

        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)

        json.dump(json_dict, open(self.dst_path, 'w'), indent=4, ensure_ascii=False)

    def __call__(self, *args, **kwargs):
        if self.mode == 'coco2voc':
            self.coco2voc()

        if self.mode == 'voc2coco':
            self.voc2coco()


if __name__ == '__main__':
    """
    how to use
    """
    # mode is coco2voc
    format_convert = FormatConverter(src_path='/Users/sunhaokai/Downloads/annotations/UAV-benchmark-M-Train.json',
                                   dst_path='/Users/sunhaokai/Downloads/UAVVOCdevkit0/VOC2007/Annotations',
                                   mode='coco2voc')
    format_convert()

    # mode is voc2coco
    # tmp = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster',
    #  'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记',
    #  '花西子', 'Clarins/娇韵诗', 'L\'occitane / 欧舒丹','Versace / 范思哲','Mizuno / 美津浓',
    #  'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰',
    #  'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED',
    #  'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀',
    #  'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe',
    #  'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克',
    #  'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano',
    #  '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻'
    #  )
    # classes = {}
    # for i,name in enumerate(tmp):
    #     classes[name] = i
    #
    # format_convert = FormatConvert(src_path='000',
    #                                dst_path='val.json',
    #                                mode='voc2coco',
    #                                classes=classes)
    # format_convert()
