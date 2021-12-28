#!/bin/bash
set -e

eval_dir='evaluation'

if [[ ! -e $eval_dir ]]; then
    mkdir $eval_dir
fi

cd darknet

for file in ../models/*.weights; do
#    echo $file
    bname="$(basename -- $file)"
    text="${bname/'weights'/'txt'}"
    ./darknet detector map ./cfg/template.data ./cfg/template.cfg $file > ../$eval_dir/$text
done