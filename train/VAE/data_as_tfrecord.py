#!/usr/bin/python
# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format

The image data set is expected to reside in JPEG files ends up with '.jpg'.

This script converts the train and testing data into
a sharded data set consisting of TFRecord files
  train_directory/train-00000-of-00128
  train_directory/train-00001-of-00128
  ...
  train_directory/train-00127-of-00128
and
  test_directory/test-00000-of-00128
  test_directory/test-00001-of-00128
  ...
  test_directory/test-00127-of-00128
where we have selected 128 shards for both data sets. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'
  image/filename: string, the unique id of the image file
            e.g. '97c0a12e07ae8dd5' or '650c989dd3493748'
Furthermore, if the data set type is train, it would contain one more field:
  image/class/label: integer, the landmark_id from the input train csv file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import tqdm

from algorithms.io.path_definition import get_project_dir
from train.VAE.utils_default_parameters import get_tf_record_default_parameters

default_parameters = get_tf_record_default_parameters()

FLAGS = flags.FLAGS

flags.DEFINE_string('train_directory', default_parameters['TRAIN_DIRECTORY'], 'Training data directory.')
flags.DEFINE_string('val_directory', default_parameters['VAL_DIRECTORY'], 'Validation data directory.')
flags.DEFINE_string('test_directory', default_parameters['TEST_DIRECTORY'], 'Testing data directory.')
flags.DEFINE_string('output_directory', default_parameters['OUTPUT_DIRECTORY'], 'Output data directory.')
flags.DEFINE_string('train_csv_path', default_parameters['TRAIN_CSV_PATH'],
                    'Training data csv file path.')
flags.DEFINE_string('val_csv_path', default_parameters['VAL_CSV_PATH'],
                    'Validation data csv file path.')
flags.DEFINE_string('test_csv_path', default_parameters['TEST_CSV_PATH'],
                    'Testing data csv file path.')
flags.DEFINE_integer('num_shards', default_parameters['NUM_SHARDS'], 'Number of shards in output data.')
flags.DEFINE_integer('image_size', default_parameters['IMAGE_SIZE'], 'Size of image')


def _get_image_files_and_conditions(name, csv_path, image_dir):
    """Process input and get the image file paths, image ids and the labels.

    Args:
        name: 'train' or 'val' or 'test'.
        csv_path: path to the csv Data Sources files.
        image_dir: directory that stores downloaded images.

    Returns:
        image_paths: the paths to all images in the image_dir.
        file_ids: the unique ids of images.
        labels: the landmark id of all images. When name='test', the returned labels
          will be an empty list.
    Raises:
        ValueError: if input name is not supported.
    """

    dir_train = f"{get_project_dir()}/data/train"

    labels_drop = []

    if name in ['train', 'val', 'test']:
        with tf.io.gfile.GFile(csv_path, 'rb') as csv_file:
            df = pd.read_csv(csv_file).drop(labels=labels_drop, axis=1)
        df = df.set_index('image_id')
        image_paths = [os.path.join(dir_train, image_dir, index) for index in df.index]
        file_ids = [index[:-4] for index in df.index]
        conditions = [df.loc[index].to_dict() for index in tqdm(df.index)]
    else:
        raise ValueError('Unsupported dataset split name: %s' % name)

    return image_paths, file_ids, conditions


def _process_image(filename):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.jpg'.

    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    Raises:
        ValueError: if parsed image has wrong number of dimensions or channels.
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = tf.io.decode_jpeg(image_data, channels=3)

    # Check that image converted to RGB
    if len(image.shape) != 3:
        raise ValueError('The parsed image number of dimensions is not 3 but %d' %
                         (image.shape))
    height = FLAGS.image_size
    width = FLAGS.image_size
    if image.shape[2] != 3:
        raise ValueError('The parsed image channels is not 3 but %d' %
                        (image.shape[2]))

    return image_data, height, width


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_id, image_buffer, height, width, condition=None):
    """Build an Example proto for the given inputs.

    Args:
        file_id: string, unique id of an image file, e.g., '97c0a12e07ae8dd5'.
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
        label: integer, the landmark id and prediction label.

    Returns:
        Example proto.
    """

    c = []
    if condition is not None:
        c = [(k, v) for k, v in condition.items()]
        c.sort()
        c = [v.encode('utf-8') for _, v in c]

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    features = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format.encode('utf-8')),
        'image/id': _bytes_feature(file_id.encode('utf-8')),
        'image/encoded': _bytes_feature(image_buffer),
        'image/class/conditions': tf.train.Feature(bytes_list=tf.train.BytesList(value=c))
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def _write_tfrecord(output_prefix, image_paths, file_ids, conditions):
    """Read image files and write image and label data into TFRecord files.

    Args:
        output_prefix: string, the prefix of output files, e.g. 'train'.
        image_paths: list of strings, the paths to images to be converted.
        file_ids: list of strings, the image unique ids.
        labels: list of integers, the landmark ids of images. It is an empty list
          when output_prefix='test'.

    Raises:
        ValueError: if the length of input images, ids and labels don't match
    """
    if not len(image_paths) == len(file_ids) == len(conditions):
        raise ValueError('length of image_paths, file_ids, labels shoud be the' +
                         ' same. But they are %d, %d, %d, respectively' %
                         (len(image_paths), len(file_ids), len(conditions)))

    spacing = np.linspace(0, len(image_paths), FLAGS.num_shards + 1, dtype=np.int)

    _, _, dir_train = get_app_dir()

    if not os.path.isdir(os.path.join(dir_train, FLAGS.output_directory)):
        os.makedirs(os.path.join(dir_train, FLAGS.output_directory))

    for shard in range(FLAGS.num_shards):
        output_file = os.path.join(os.path.join(dir_train, FLAGS.output_directory),
                                   '%s-%.5d-of-%.5d' % (output_prefix, shard, FLAGS.num_shards))
        writer = tf.io.TFRecordWriter(output_file)
        print('Processing shard ', shard, ' and writing file ', output_file)
        for i in range(spacing[shard], spacing[shard + 1]):
            image_buffer, height, width = _process_image(image_paths[i])
            example = _convert_to_example(file_ids[i], image_buffer, height, width,
                                    conditions[i])
            writer.write(example.SerializeToString())
        writer.close()


def _build_tfrecord_dataset(name, csv_path, image_dir):
    """Build a TFRecord dataset.

    Args:
        name: 'train' or 'test' to indicate which set of data to be processed.
        csv_path: path to the Google-landmark Dataset csv Data Sources files.
        image_dir: directory that stores downloaded images.

    Returns:
        Nothing. After the function call, sharded TFRecord files are materialized.
    """

    dir_train = f"{get_project_dir()}/data/train"

    image_paths, file_ids, conditions = _get_image_files_and_conditions(name, os.path.join(dir_train, csv_path),
                                                                        image_dir)
    _write_tfrecord(name, image_paths, file_ids, conditions)


def main(unused_argv):

    _build_tfrecord_dataset('train', FLAGS.train_csv_path, FLAGS.train_directory)
    _build_tfrecord_dataset('val', FLAGS.val_csv_path, FLAGS.val_directory)
    _build_tfrecord_dataset('test', FLAGS.test_csv_path, FLAGS.test_directory)


if __name__ == '__main__':
    app.run(main)