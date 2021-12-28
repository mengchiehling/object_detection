#!/bin/bash
set -e

d=2021-11-29  # [fashion] the s3 index of data that will be used for training
pretrained_model=darknet53.conv.74 # [fashion] pretrained weights

# Download data from AWS.
aws s3 sync s3://c24-fa-ds-object-detection/data/train/datasets/YOLO/training/$d ./data
# official weight source is https://pjreddie.com/media/files/darknet53.conv.74, we saved it in c24-fa-ds-object-detection/data/train/models/YOLO
aws s3 cp s3://c24-fa-ds-object-detection/data/train/models/YOLO/$pretrained_model ./darknet/

# replace ~ with current folder
sed -i 's:~:'`pwd`':' example/template.data

# prepare the location of trained models, according to your backup in .data
if [ -d "models" ]; then rm -Rf models; fi
mkdir models

# training data book keepin g
cd data
find $PWD -name '*train*.jpg' > train.txt
find $PWD -name '*val*.jpg' > test.txt

# compile
cd ../darknet
make