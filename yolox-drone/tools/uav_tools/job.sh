#!/bin/sh
#JSUB -J visdrone-train
#JSUB -n 6
#JSUB -q gpu
#JSUB -o out/out.%J
#JSUB -e err/err.%J
#JSUB -m h3cgpu01
source /public/jhinno/unischeduler/job_starter/unisched
################################################
# $JH_NCPU: Number of CPU cores #
# $JH_HOSTFILE: List of computer hostfiles #
################################################
module load miniconda3
module load cuda/11.1
source activate mmdet
export  LD_LIBRARY_PATH=/public/system/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0
python uavformat_converter.py