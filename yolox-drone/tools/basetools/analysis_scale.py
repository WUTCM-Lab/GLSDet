import os
import glob
import torch
import sys
import xml.etree.ElementTree as ET


def cal_scale_parameter_per_image(inputs, num_classes=10):
    """
        args:
            inputs :[boxes_num, 5]
                    5 means {corner_x, corner_y, width, height, class}
            num_classes: the total number of classes of datasets

        return:
            [num_classes, 2]
                    2 means {sum of scale parameter, number of boxes}
    """

    outputs = torch.zeros([num_classes, 2])
    for cls in range(num_classes):
        index = torch.where(inputs[:, 4] == cls)
        w_h = inputs[index][:, 2:4]

        areas, _ = torch.prod(w_h, dim=1).sort(descending=False)
        if areas.shape[0] == 0:
            outputs[cls][0] = 0
            outputs[cls][1] = 0
            continue

        areas_copy = torch.zeros_like(areas)
        areas_copy[1:] = areas[:-1]
        areas_copy[0] = areas[0]

        si = (areas * 1.0 / (areas_copy * 1.0)).sum()
        boxes_num = areas.size()[0]
        outputs[cls][0] = boxes_num
        outputs[cls][1] = si

        """
        display intermediate results
        """
        print("cls_{}:(boxes_num:{},sum:{:.4f})".format(cls, boxes_num, si))
    print(outputs, '\n')

    return outputs


def read_xml(src_path, classes={'pedestrian': 0, 'people': 1, 'bicycle': 2, 'car': 3,
                                'van': 4, 'truck': 5, 'tricycle': 6, 'awning-tricycle': 7, 'bus': 8, 'motor': 9}):
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

    all_boxes_info = []
    xml_paths = glob.glob(os.path.join(src_path, '*.xml'))
    for image_id, xml_f in enumerate(xml_paths):
        boxes_info = []
        sys.stdout.write('\r>> Converting file %d/%d' % (
            image_id + 1, len(xml_paths)))
        sys.stdout.flush()

        tree = ET.parse(xml_f)
        root = tree.getroot()

        for obj in get(root, 'object'):
            cls = get_and_check(obj, 'name', 1).text
            if cls not in classes.keys():
                continue

            bbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bbox, 'xmin', 1).text)
            ymin = int(get_and_check(bbox, 'ymin', 1).text)
            xmax = int(get_and_check(bbox, 'xmax', 1).text)
            ymax = int(get_and_check(bbox, 'ymax', 1).text)
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes_info.append([(xmin+xmax)/2.0, (ymin+ymax)/2.0, xmax-xmin, ymax-ymin, classes[cls]])
        all_boxes_info.append(boxes_info)

    return all_boxes_info


if __name__ == '__main__':
    """
    test data
    """
    # inputs1 = torch.tensor([[0, 1, 2, 100, 1], [0, 1, 4, 6, 1], [0, 1, 3, 5, 1], [0, 1, 2, 10, 1], [0, 1, 10, 6, 1], [0, 1, 3, 5, 1],
    #                       [0, 1, 2, 5, 0], [0, 1, 4, 6, 0], [0, 1, 3, 5, 0]])
    #
    # inputs2 = torch.tensor(
    #     [[0, 1, 2, 100, 1], [0, 1, 4, 6, 1], [0, 1, 3, 5, 1], [0, 1, 2, 10, 1], [0, 1, 10, 6, 1], [0, 1, 3, 5, 1],
    #      [0, 1, 2, 5, 0], [0, 1, 4, 6, 0], [0, 1, 3, 5, 0]])
    # images = [inputs1, inputs2]

    classes_path = '/home/server2/shk/yolox_drone/model_data/visdrone10.txt'
    xml_dir = '/home/server2/shk/yolox_drone/VOCdevkit_val/VOC2007/Annotations'

    f = open(classes_path)
    classes = {}
    id = 0
    for line in f:
        classes[line.strip()] = id
        id += 1
    print(classes)
    images = read_xml(xml_dir, classes=classes)

    num_classes = len(classes)
    final_results = torch.zeros([10, 2])

    for inputs in images:
        inputs = torch.tensor(inputs)
        results = cal_scale_parameter_per_image(inputs=inputs, num_classes=num_classes)
        final_results += results
    si_value = final_results[:, 1] / final_results[:, 0]

    print(final_results)
    print(si_value)
