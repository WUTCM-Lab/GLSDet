import os
import glob
import torch
import sys
import xml.etree.ElementTree as ET

SCALE_RATIO = 1
IS_RATIO = False

def cal_object_parameter_per_image(inputs, num_classes=10):
    """
        args:
            inputs :[boxes_num, 7]
                    5 means {corner_x, corner_y, width, height, class, image_width, image_height}
            num_classes: the total number of classes of datasets

        return:
            [num_classes, 2]
                    2 means {sum of object area, number of boxes}
    """

    outputs = torch.zeros([num_classes, 4])
   #  image_size = inputs[0][5] * inputs[0][6]
    for cls in range(num_classes):
        if  inputs.shape[0] == 0:
            continue

        index = torch.where(inputs[:, 4] == cls)
        w_h = inputs[index][:, 2:4]
        areas = torch.prod(w_h, dim=1)
        boxes_num = areas.size()[0]

        if IS_RATIO:
            criterion = [0, 0.01, 0.1]
        else:
            criterion = [0, 32.0 ** 2, 96 ** 2]
        small_index = torch.where(areas <= criterion[1])
        large_index = torch.where(areas > criterion[2])

        if IS_RATIO:
            outputs[cls][0] = areas.sum() / SCALE_RATIO / 1
        else:
            outputs[cls][0] = areas.sum() / SCALE_RATIO
        outputs[cls][1] = boxes_num
        outputs[cls][2] = torch.tensor(small_index[0].size()[0])
        outputs[cls][3] = torch.tensor(large_index[0].size()[0])
        """
        display intermediate results
        """
    #     print("cls_{}:(boxes_num:{},sum:{:.4f})".format(cls, boxes_num, areas.sum() / (boxes_num * 1.0)))
    # print(outputs, '\n')

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

        for obj in get(root, 'size'):
            width = get_and_check(obj, 'width', 1).text
            height = get_and_check(obj, 'height', 1).text

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

            boxes_info.append([(xmin+xmax)/2.0, (ymin+ymax)/2.0, xmax-xmin, ymax-ymin, classes[cls], int(width), int(height)])
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

    classes_path = '/home/server2/shk/yolox_drone/model_data/uav3.txt'
    xml_dir = '/home/server2/shk/yolox_drone/0221trainufp/VOC2007/Annotations'

    f = open(classes_path)
    classes = {}
    id = 0
    for line in f:
        classes[line.strip()] = id
        id += 1
    print(classes)
    images = read_xml(xml_dir, classes=classes)

    num_classes = len(classes)
    final_results = torch.zeros([len(classes), 4])

    for inputs in images:
        inputs = torch.tensor(inputs)
        results = cal_object_parameter_per_image(inputs=inputs, num_classes=num_classes)
        final_results += results
    si_value = final_results[:, 0] / final_results[:, 1] * SCALE_RATIO

    # print(final_results)
    # print(si_value)
    # if IS_RATIO:
    #     criterion = [0, 0.01, 0.1]
    # else:
    #     criterion = [0, 32.0**2, 96**2]
    # criterion_message = ['small', 'medium', 'large']
    # criterion_eval = torch.ones(si_value.size())
    # small_index = torch.where(si_value < criterion[1])
    # large_index = torch.where(si_value > criterion[2])
    # criterion_eval[small_index] = 0
    # criterion_eval[large_index] = 2
    # print(si_value)
    # for i, key in enumerate(classes.keys()):
    #     print("class_{}, num:{}, size:{}, {}".format(key, final_results[i][1], round(float(si_value[i]), 5),
    #                                             criterion_message[int(criterion_eval[i])]))
    #
    # for i, key in enumerate(classes.keys()):
    #     print("class_{}, small_{}, medium_{}, large_{}".format(key, final_results[i][2],
    #                                                            final_results[i][1] - final_results[i][2],
    #                                                            final_results[i][1]))

    print(final_results[:, 2].sum())
    print((final_results[:, 1] - final_results[:, 2] - final_results[:, 3]).sum())
    print(final_results[:, 3].sum())

