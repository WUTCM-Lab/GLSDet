import re
import torch
from collections import OrderedDict


def init_train(src_model_path, dts_model_path, epoch=0, iter=0):
    ckpt_dict = torch.load(src_model_path)
    ckpt_dict['meta']['epoch'] = epoch
    ckpt_dict['meta']['iter'] = iter
    torch.save(ckpt_dict, dts_model_path)


def paras_add_BN_running(src_file, dst_file=None):
    """
    for EMA
    """
    if dst_file is None:
        dst_file = src_file.replace('.txt', '_plus.txt')

    f_src = open(src_file, 'r')
    f_dst = open(dst_file, 'w')
    for line in f_src:
        f_dst.write(line.strip('\n') + '\n')

        str_paras = re.findall('.*bn.bias', line.strip('\n'))
        if len(str_paras) != 0:
            str_size = re.findall('(torch.Size.*)', line.strip('\n'))[0]
            str_para = str_paras[0]
            f_dst.write(str_para.replace('.bias', '.running_mean') + ' ' + str_size + '\n')
            f_dst.write(str_para.replace('.bias', '.running_var') + ' ' + str_size + '\n')
            f_dst.write(str_para.replace('.bias', '.num_batches_tracked') + ' ' + 'torch.Size([])' + '\n')

    f_dst.close()
    f_src.close()


def get_ckpt_paras_info(ckpt_file, output_file='', mode='default'):
    """
    if the frame is mmdetection, please set mode='mmdet',
    the checkpoint is consisted of dict{'meta':,
    'state_dict':collections.OrderedDict()}

     meta = dict{
        env_info <class 'str'>
        config <class 'str'>
        seed <class 'int'>
        exp_name <class 'str'>
        hook_msgs <class 'dict'>
        epoch <class 'int'> !important
        iter <class 'int'> !important
        mmcv_version <class 'str'>
        time <class 'str'>
        CLASSES <class 'tuple'> }

    """

    paras_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

    if mode == 'mmdet':
        for k, v in paras_dict['meta'].items():
            print(k, v)
        paras_dict = paras_dict['state_dict']

    elif mode == 'megvii':
        paras_dict = paras_dict['model']

    f = open(output_file, 'w')
    for k, v in paras_dict.items():
        print(k)
        f.write(k + ' ' + str(v.size()) + '\n')
    f.close()


def get_para_name_and_shape(line_str):
    para = line_str.split(' ')[0]

    shapes = re.findall('torch.Size\(\[([0-9].*)\]\)', line_str)

    if len(shapes) == 0:
        shapes = []
    else:
        shapes = [int(i) for i in shapes[0].split(',')]
    shapes = torch.Size(shapes)

    return para, shapes


def convert_ckpt_mmdet(paras_name_file, src_paras_name_file, src_ckpt, dst_ckpt=None, ema=False, mode='default'):
    """
    args:
        paras_name_file: the file_path to the new checkpoint's parameters' names
        src_paras_name_file: the file path to the original checkpoint
        Is is important in the name_file, one to one correctly!!!

        ema: if use the ema strategy,it is suggested to set True
    """

    if dst_ckpt is None:
        dst_ckpt = src_ckpt.replace('.pth', '_new' + '.pth')
        print(dst_ckpt)

    f_paras_name = open(paras_name_file, 'r')
    f_src_paras_name = open(src_paras_name_file, 'r')
    if mode == 'mmdet':
        src_paras_dict = torch.load(src_ckpt, map_location=torch.device('cpu'))['state_dict']
    else:
        src_paras_dict = torch.load(src_ckpt, map_location=torch.device('cpu'))

    dst_dict = {}
    dst_dict['meta'] = {}
    dst_dict['meta']['epoch'] = 0
    dst_dict['meta']['iter'] = 0
    dst_dict['state_dict'] = OrderedDict()
    dst_paras_dict = dst_dict['state_dict']

    while True:
        line_dst = f_paras_name.readline().strip('\n')
        line_src = f_src_paras_name.readline().strip('\n')
        if line_src.strip('\n') == '':
            break
        src_para, src_shapes = get_para_name_and_shape(line_src)
        dst_para, dst_shapes = get_para_name_and_shape(line_dst)
        if src_shapes != dst_shapes:
            raise ValueError(src_para, src_shapes, dst_para, dst_shapes)

        print(src_para + '>>>>>>>' + dst_para)
        dst_paras_dict[dst_para] = src_paras_dict[src_para]
        if ema:
            dst_paras_dict[('ema_' + dst_para).replace('.', '_')] = src_paras_dict[src_para]

    f_paras_name.close()
    torch.save(dst_dict, dst_ckpt)


def convert_ckpt(paras_name_file, src_paras_name_file, src_ckpt, dst_ckpt=None, ema=False, mode='default'):
    """
    args:
        paras_name_file: the file_path to the new checkpoint's parameters' names
        src_paras_name_file: the file path to the original checkpoint
        Is is important in the name_file, one to one correctly!!!

        ema: if use the ema strategy,it is suggested to set True
    """

    if dst_ckpt is None:
        dst_ckpt = src_ckpt.replace('.pth', '_new' + '.pth')
        print(dst_ckpt)

    f_paras_name = open(paras_name_file, 'r')
    f_src_paras_name = open(src_paras_name_file, 'r')
    if mode == 'mmdet':
        src_paras_dict = torch.load(src_ckpt, map_location=torch.device('cpu'))['state_dict']
    else:
        src_paras_dict = torch.load(src_ckpt, map_location=torch.device('cpu'))

    dst_dict = OrderedDict()

    while True:
        line_dst = f_paras_name.readline().strip('\n')
        line_src = f_src_paras_name.readline().strip('\n')
        if line_src.strip('\n') == '':
            break
        src_para, src_shapes = get_para_name_and_shape(line_src)
        dst_para, dst_shapes = get_para_name_and_shape(line_dst)
        if src_shapes != dst_shapes:
            raise ValueError(src_para, src_shapes, dst_para, dst_shapes)

        print(src_para + '>>>>>>>' + dst_para)
        dst_dict[dst_para] = src_paras_dict[src_para]
        if ema:
            dst_dict[('ema_' + dst_para).replace('.', '_')] = src_paras_dict[src_para]

    f_paras_name.close()
    torch.save(dst_dict, dst_ckpt)


def check_checkpoints(mmdet_file, file, mmdet_paras_file):
    dict_1 = torch.load(mmdet_file)['state_dict']
    dict_2 = torch.load(file)
    f = open(mmdet_paras_file, 'r')
    for _, v in dict_2.items():
        line_src = f.readline().strip('\n')
        if line_src.strip('\n') == '':
            break

        para, _ = get_para_name_and_shape(line_src)
        if not dict_1[para].equal(v):
            print(para)

    f.close()


def check_parameters_name_align(fiel1, fiel2, isName=False):
    f1 = open(fiel1, 'r')
    f2 = open(fiel2, 'r')
    index = 0
    flag = True
    while True:
        index += 1
        line_1 = f1.readline().strip('\n')
        line_2 = f2.readline().strip('\n')
        if line_1.strip('\n') == '' or line_2.strip('\n') == '':
            break
        p1, s1 = get_para_name_and_shape(line_1)
        p2, s2 = get_para_name_and_shape(line_2)
        if s1 != s2 or (p1 != p2 and isName):
            flag = False
            print(str(index) + ":" + "key:{},value:{}>>>>>key:{},value:{}".format(p1, s1, p2, s2))

    if flag:
        print('all right!')


if __name__ == '__main__':
    src_para_names = 'txt/mmdet_yolox_m_backbone.txt'
    src_ckpt = 'model_data/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    dst_ckpt = ''

    # init_train('/Users/sunhaokai/Desktop/yolox_l_8x8_300e_coco_20211126.pth', 'mmdet_yolox_l.pth')

    '''
    * not necessary, help you check parameters
    first step:
        process the original checkpoint, get the parameters_info_file
    '''
    # get_ckpt_paras_info('/Users/sunhaokai/Desktop/yolox_m.pth', 'tmp/megvii.txt', mode='megvii')
    get_ckpt_paras_info('/home/server2/shk/yolox_drone/work_dir/baseline640/final152-loss3.885-val_loss4.041.pth', '111.txt')
    # check_parameters_name_align('tmp/bb.txt', 'tmp/megvii.txt', isName=True)

    '''
    * not necessary, especially for mmdetection EMA scheme
    second step:
        add some parameters
    '''
    # paras_add_BN_running('tmp/mmdet_m_src.txt', 'tmp/mmdet_m_model.txt')

    '''
    third step:
       convert checkpoint
    '''
    # convert_ckpt('tmp2.txt', 'tmp0.txt',
    #              '../model_data/yolox_m.pth',
    #              '../model_data/yolox_m_new.pth', ema=False)

    '''
    fourth step:
       check checkpoint
    '''
    # check_checkpoints('model_data/mmdet_yolox_m_backbone.pth', 'model_data/yolox_m.pth', 'txt/mmdet_yolox_m_backbone.txt')
