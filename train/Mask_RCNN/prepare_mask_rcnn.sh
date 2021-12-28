#!/bin/bash

# Download the data you need for training/validation
aws s3 sync s3://c24-fa-ds-object-detection/data/train/datasets/Mask_RCNN/training/2021-07-28/ data # [fashion] download data marked bu 2021-07-28 in s3

# Work-around to have "conda" commands available
CONDA_PREFIX=~/anaconda3             # adjust manually
source $CONDA_PREFIX/etc/profile.d/conda.sh

# Create the working environment
conda env create -f environment.yml --force --name mask-rcnn