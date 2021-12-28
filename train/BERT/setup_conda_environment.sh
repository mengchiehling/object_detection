#!/usr/bin/env bash

# CAUTION: Ensure that you are in project directory

ENV_NAME=bert         # should be the repo name
CONDA_PREFIX=~/miniconda3               # adjust manually

# Work-around to have "conda" commands available
source $CONDA_PREFIX/etc/profile.d/conda.sh

conda env create -f environment.yml --force --name $ENV_NAME

conda activate $ENV_NAME