# GLSDet
<p align="center"> <img src='./model.jpg' align="center" height="540px"> </p>

This is the official implementation of [**Global–Local Fusion With Semantic Information Guidance for Accurate Small Object Detection in UAV Aerial Images**](https://ieeexplore.ieee.org/document/10849626). This paper has been accepted by TGRS 2025.

    @ARTICLE{10849626,
      author={Chen, Yaxiong and Ye, Zhengze and Sun, Haokai and Gong, Tengfei and Xiong, Shengwu and Lu, Xiaoqiang},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Global–Local Fusion With Semantic Information Guidance for Accurate Small Object Detection in UAV Aerial Images}, 
      year={2025},
      volume={63},
      number={},
      pages={1-15},
      keywords={Object detection;Autonomous aerial vehicles;Feature extraction;Semantics;Detectors;Accuracy;Technological innovation;Superresolution;Assembly;Transformers;Decoupled head attention;feature fusion;remote sensing image recognition;robust adversarial;robustness;rotational object detection},
      doi={10.1109/TGRS.2025.3532612}}
}

### Installation
1.  Clone this repository.
    ```
    git clone https://github.com/LearnYZZ/GLSDet
    ```

2.  Prepare for the running environment. 

    ```
    pip install -e .
    ```

# Install

1. This repo is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install it according to [get_start.md](docs/en/get_started.md).

2. ```shell
   pip install nltk
   pip install albumentations
   ```

## Quickstart

We provide the Dataset(COCO Format) as follows:

- VisDrone:链接：https://pan.baidu.com/s/1FfAsAApHZruucO5A2QgQAg 提取码：qrvs
- UAVDT:链接：链接：https://pan.baidu.com/s/1KLmU5BBWwgtFbuZa7QWavw 提取码：z08x

We provide the checkpoint as follows:

- VisDrone Coarse-Det:链接: https://pan.baidu.com/s/1jK3bqImDGSwqRJGVXinS0w 提取码: nab3
- VisDrone MP-Det ResNet50: 链接: https://pan.baidu.com/s/1zOoJVO2fPejnzM9KioZLuQ 提取码: m7rj

# Training

This repo is only supposed single GPU.

## Prepare

Build by yourself: We provide two data set conversion tools.

```shell
# conver VisDrone to COCO
python UFPMP-Det-Tools/build_dataset/VisDrone2COCO.py
# conver UAVDT to COCO
python UFPMP-Det-Tools/build_dataset/UAVDT2COCO.py
# build UFP dataset(VisDrone)
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    xxxxxx/dataset/COCO/images/UAVtrain \
    xxxxxx/dataset/COCO/annotations/instances_UAVtrain_v1.json \
    xxxxxx/dataset/COCO/images/instance_UFP_UAVtrain/ \
    xxxxxx/dataset/COCO/annotations/instance_UFP_UAVtrain.json \
    --txt_path path_to_VisDrone_annotation_dir
```

Download:

In Quick Start

## Train Coarse Detector

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/UFPMP-Det/coarse_det.py
```

## Train MP-Det

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./config/UFPMP-Det/mp_det_res50.py
```

# Test

```shell
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res50.py  \
    ./work_dirs/mp_det_res50/epoch_12.pth \
    XXXXX/dataset/COCO/annotations/instances_UAVval_v1.json \
    XXXXX/dataset/COCO/images/UAVval

```

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@inproceedings{ufpmpdet,
  title={UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery},
  author={Huang, Yecheng and Chen, Jiaxin and Huang, Di},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

