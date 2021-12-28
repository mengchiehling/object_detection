#!/bin/bash
set -e

d=$1  # [fashion] the index in AWS S3, must be user specified

echo "upload key folder is $d"

aws s3 sync fashion/models/ s3://c24-fa-ds-object-detection/data/train/models/YOLO/$d/models   # [fashion] upload your models to AWS S3