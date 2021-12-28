from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from copy import copy

import tensorflow as tf

from algorithms.io.path_definition import get_project_dir

dir_train = f"{get_project_dir()}/data/train"


class DataAugmentationParams(object):
    """Default parameters for augmentation."""
    # The following are used for train.

    random_reflection = True
    random_contrast = True
    random_saturation = True


def image_net_crop(image, image_height, image_width):
    """Imagenet-style crop with random bbox and aspect ratio.

    Args:
    image: a `Tensor`, image to crop.

    Returns:
    cropped_image: `Tensor`, cropped image.
    """

    params = DataAugmentationParams()
    cropped_image = tf.dtypes.cast(copy(image), tf.int32)

    if params.random_reflection:
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    if params.random_contrast:
        cropped_image = tf.image.random_contrast(cropped_image, lower=0.8, upper=1)
        cropped_image = tf.clip_by_value(cropped_image, 0, 255)
    if params.random_saturation:
        cropped_image = tf.image.random_saturation(cropped_image, lower=0.9, upper=1.1)
        cropped_image = tf.clip_by_value(cropped_image, 0, 255)

    # normalize the image to range 0-1
    cropped_image = tf.image.resize(cropped_image, [image_height, image_width])
    cropped_image = tf.math.divide(cropped_image, 255)
    return cropped_image


def parse_function(example, name_to_features, image_height, image_width, augmentation, extra_condition):
    """Parse a single TFExample to get the image and label and process the image.

    Args:
    example: a `TFExample`.
    name_to_features: a `dict`. The mapping from feature names to its type.
    image_size: an `int`. The image size for the decoded image, on each side.
    augmentation: a `boolean`. True if the image will be augmented.

    Returns:
    image: a `Tensor`. The processed image.
    label: a `Tensor`. The ground-truth label.
    """
    parsed_example = tf.io.parse_single_example(example, name_to_features)
    # Parse to get image.
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    if augmentation:
        image = image_net_crop(image, image_height, image_width)
    else:
        image = tf.dtypes.cast(copy(image), tf.int32)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.math.divide(image, 255)

    # Parse to get image attributes.
    if extra_condition:
        conditions = parsed_example['image/class/attributes']
        return image, conditions
    else:
        return image


def create_dataset(file_pattern,
                   image_height,
                   image_width,
                   batch_size=32,
                   augmentation=False,
                   seed=0):
    """Creates a dataset.

    Args:
        file_pattern: str, file pattern of the dataset files.
        image_size: int, image size.
        batch_size: int, batch size.
        augmentation: bool, whether to apply augmentation.
        seed: int, seed for shuffling the dataset.

    Returns:
        tf.data.TFRecordDataset.
    """

    filenames = tf.io.gfile.glob(file_pattern)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat().shuffle(buffer_size=1024, seed=seed)

    # Create a description of the features.
    feature_description = {
      'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/class/conditions': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True)}

    customized_parse_func = functools.partial(
      parse_function,
      name_to_features=feature_description,
      image_height=image_height,
      image_width=image_width,
      augmentation=augmentation,
      extra_condition=False)
    dataset = dataset.map(customized_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset
