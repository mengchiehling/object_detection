#!/usr/bin/env bash

# WARNING Carefully check all variable settings here

# Absolute path to the original git repo
PROJECT=~/<folder name>
CONDA_PREFIX=~/miniconda3
# The branch to package
# Absolute path to the directory where all package files are gathered
# The `source > path` field in the meta.yaml file in the conda recipe has to point to the build directory

# clone remote project
#git clone --branch $BRANCH $PROJECT --depth 1 $BUILD_DIR &&
# clone local project. The repo should be commited
#git clone $PROJECT $BUILD_DIR
cp -r $PROJECT/config $PROJECT/vs_package/config
cp -r $PROJECT/data $PROJECT/vs_package/data

## Start the conda build process
conda install -c conda-forge conda-build -y
conda build $PROJECT/build/conda_recipe/ -c defaults -c conda-forge

cp $CONDA_PREFIX/conda-bld/noarch/vs-package-1.0-py_0.tar.bz2 ~/datascience-conda-packages/conda_channel/noarch/
conda index --verbose ~/datascience-conda-packages/conda_channel/

rm -r $PROJECT/vs_package/config
rm -r $PROJECT/vs_package/data