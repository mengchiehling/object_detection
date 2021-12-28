#!/bin/bash
set -e

# you need to change the file names of cfg and data mannually if you do not want to type the last line.
cfg=template.cfg # [fashion] neural configurations
data=template.data # [fashion] model metadata
pretrained_model=darknet53.conv.74 # [fashion] pretrained weight, should be consistent with network specified by fashion.cfg

# start trainingprepare_yolo.sh
cd darknet
./darknet detector train ./cfg/$data ./cfg/$cfg ./$pretrained_model -dont_show > train.log # [fashion] execution