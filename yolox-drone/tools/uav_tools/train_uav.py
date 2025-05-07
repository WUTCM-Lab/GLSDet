import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from models.core.callbacks import LossHistory
from models.core.dataloader import YoloDataset, yolo_dataset_collate
from models.core.utils import get_classes
from models.core.utils_fit import fit_one_epoch

# from models.base.yolox import YoloBody
from models.decouple.yolox_recls import YoloBody
# from models.cfp.yolox_cfp import YoloBody

# from models.base.yolox_losses import YOLOLoss, weights_init
# from models.losses.yolox_losses_no_sigmoid import YOLOLoss, weights_init
# from models.losses.yolox_losses_fpn_weight import YOLOLoss, weights_init
from models.losses.yolox_losses_fpn_weight_sigmoid640 import YOLOLoss, weights_init


def setup_seed(seed=4400):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        #os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    seed_num = int(random.random() * 40000)
    print('seed is {}'.format(seed_num))
    setup_seed(seed=seed_num)

    input_shape = [640, 640]
    ONLY_BACKBONE = False
    Cuda = True
    IF160epoch = False

    # model_path = 'model_data/yolox_m.pth'
    model_path = 'work_dir/uav_recls_80_20831/ep040-loss1.830-val_loss1.960.pth'
    log_dir = 'work_dir/uav_recls40_loss40' + '_' + str(seed_num)
    # log_dir = 'work_dir/tmp'

    # ------------------------------------------------------#
    #   所使用的YoloX的版本。nano、tiny、s、m、l、x
    # ------------------------------------------------------#
    phi = 'm'
    # --------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # --------------------------------------------------------#
    classes_path = '/share/home/wut_zhangb/UAVDT/classes.txt'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------------------------------------------------#
    #   YoloX的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   YOLOX作者强调要在训练结束前的N个epoch关掉Mosaic。因为Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #   并且Mosaic大量的crop操作会带来很多不准确的标注框，本代码自动会在前90%个epoch使用mosaic，后面不使用。
    #   Cosine_scheduler 余弦退火学习率 True or False
    # ------------------------------------------------------------------------------------------------------------#
    mosaic = False
    scheduler_mode_list = ['CosineAnnealingLR', 'StepLR', 'CosineAnnealingWarmRestarts']
    scheduler_mode = scheduler_mode_list[1]

    # ----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ----------------------------------------------------#
    # ----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    # ----------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 40
    Freeze_batch_size = 4
    Freeze_lr = 1e-4
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 40
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = False
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    # ------------------------------------------------------#
    num_workers = 4
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '/share/home/wut_zhangb/UAVDT/2007_train.txt'
    val_annotation_path = '/share/home/wut_zhangb/UAVDT/2007_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)

        # pretrained_dict = pretrained_dict['model']

        pre_dict = OrderedDict()
        mismatched_k = []
        for k, v in pretrained_dict.items():
            if ONLY_BACKBONE:
                if k.find('backbone.backbone') == -1:
                    continue
            if k in model_dict:
                if np.shape(model_dict[k]) == np.shape(v):
                    pre_dict[k] = v
                    print(k)
                else:
                    mismatched_k.append(k)
            else:
                mismatched_k.append(k)
                break

        print("\n")
        for k in mismatched_k:
            print("{} is not be matched".format(k))

        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(num_classes)

    loss_history = LossHistory(log_dir)

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if scheduler_mode == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        elif scheduler_mode == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        elif scheduler_mode == 'CosineAnnealingWarmRestarts':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5,
                                                                          last_epoch=-1)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                  train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(log_dir, model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if scheduler_mode == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        elif scheduler_mode == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        elif scheduler_mode == 'CosineAnnealingWarmRestarts':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0,
                                                                          last_epoch=-1)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                  train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(log_dir, model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if IF160epoch:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = UnFreeze_Epoch
        end_epoch = UnFreeze_Epoch + 40

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if scheduler_mode == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        elif scheduler_mode == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        elif scheduler_mode == 'CosineAnnealingWarmRestarts':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0,
                                                                          last_epoch=-1)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                  train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(log_dir, model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if IF160epoch:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = UnFreeze_Epoch + 40
        end_epoch = UnFreeze_Epoch + 80

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if scheduler_mode == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        elif scheduler_mode == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        elif scheduler_mode == 'CosineAnnealingWarmRestarts':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0,
                                                                          last_epoch=-1)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                  train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(log_dir, model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
