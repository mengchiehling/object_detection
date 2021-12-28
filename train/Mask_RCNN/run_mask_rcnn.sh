#!/bin/bash

CONDA_PREFIX=~/anaconda3             # adjust manually
source $CONDA_PREFIX/etc/profile.d/conda.sh

conda activate object-detection # [fashion] activate conda environment

# Start training
export PYTHONWARNINGS="ignore" # [fashion] suppress scikit-image warning
cd ..
cd ..
python -m train.Mask_RCNN.fashion.fashion train --dataset=train/Mask_RCNN/data --weights=coco --backbone=resnet101 --object_classification=object