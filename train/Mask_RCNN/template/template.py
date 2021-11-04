"""
Mask R-CNN
Train on the fashion dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: run from the command line as such:

    # Train a new model starting from pre-trained COCO weights with resnet101
    # argument object_classification specifies the annotation label you are going to use.

    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco --backbone=resnet101
    --object_classification=object

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
"""

import os
import json
import datetime

import skimage.draw
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from train.Mask_RCNN.mrcnn.config import Config
from train.Mask_RCNN.mrcnn import model as modellib, utils
from algorithms.settings import model_architecture
from algorithms.io.path_definition import get_project_dir

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
home = os.path.expanduser("~")
DEFAULT_LOGS_DIR = f"{home}/{model_architecture['backbone']}-{datetime.datetime.now().strftime('%Y-%m-%d')}"

############################################################
#  Configurations
############################################################


class FashionConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fashion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = model_architecture['fit_params'].get('STEPS_PER_EPOCH', 100)

    VALIDATION_STEPS = model_architecture['fit_params'].get('VALIDATION_STEPS', 50)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = model_architecture.get('rpn_anchor_ratio', [0.5, 1, 2])

    # optimizer setup
    CLASS_NAME = model_architecture['optimizer_params']['class_name']
    CONFIG = model_architecture['optimizer_params'][CLASS_NAME]

    def __init__(self, object_classification: str):
        self.NUM_CLASSES = 1 + len(model_architecture['classes'][object_classification])
        # NUM_CLASSES should be set before __init__()
        # otherwise there will be size mismatch in IMAGE_META_SIZE
        super(FashionConfig, self).__init__()

############################################################
#  Dataset
############################################################


class FashionDataset(utils.Dataset):

    def __init__(self, object_classification: str):

        super().__init__()

        self.object_classification = object_classification
        self.target_classes = model_architecture['classes'][object_classification]
        self.source = 'fashion'

    def load_fashion(self, dataset_dir, subset):
        """Load a subset of the Shoe dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.

        for ix, target in enumerate(self.target_classes):
            self.add_class(self.source, ix+1, target)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(os.path.abspath(dataset_dir), subset)
        assert os.path.isdir(dataset_dir), f"{dataset_dir} does not exist, exit"

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotation_file = os.path.join(dataset_dir, "via_export_json.json")
        assert os.path.isfile(annotation_file), f"{annotation_file} does not exist, exit"
        annotations = json.load(open(annotation_file))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes'] for s in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes'] for s in a['regions']]

            num_ids = [self.target_classes.index(object[self.object_classification])+1 for object in objects]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]

            self.add_image(
                self.source,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] not in [self.source]:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_ids = image_info["num_ids"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(num_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] not in self.target_classes:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, object_classification: str):
    """Train the model."""
    # Training dataset.
    dataset_train = FashionDataset(object_classification)
    dataset_train.load_fashion(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FashionDataset(object_classification)
    dataset_val.load_fashion(args.dataset, "val")
    dataset_val.prepare()

    # Callbacks
    callbacks = None
    fit_params = model_architecture['fit_params']
    if 'callbacks' in fit_params:
        callbacks_dict = fit_params['callbacks']
        callbacks = []
        if 'EarlyStopping' in callbacks_dict:
            callbacks.append(EarlyStopping(**callbacks_dict['EarlyStopping']))
        if 'ModelCheckpoint' in callbacks_dict:
            callbacks.append(ModelCheckpoint(os.path.join(DEFAULT_LOGS_DIR, "best_model.h5"),
                                             **callbacks_dict['ModelCheckpoint']))
        if 'ReduceLROnPlateau' in callbacks_dict:
            callbacks.append(ReduceLROnPlateau(**callbacks_dict['ReduceLROnPlateau']))
        if 'LearningRateScheduler' in callbacks_dict:

            epoch_burnin = callbacks_dict['LearningRateScheduler'].get('burn_in', 1)

            def scheduler(epoch):

                optimizer_params = model_architecture['optimizer_params']

                class_name = optimizer_params['class_name']
                lr = optimizer_params[class_name]['lr']
                if epoch < epoch_burnin:
                    print(f'learning rate lr = {lr/epoch_burnin * (epoch + 1)}')
                    return lr/epoch_burnin * epoch

                steps = callbacks_dict['LearningRateScheduler']['steps']
                scaler = callbacks_dict['LearningRateScheduler']['scalers']

                for istep, step in enumerate(steps):
                    if epoch >= step:
                        lr = lr * scaler[istep]
                print(f'learning rate lr = {lr}')
                return lr
            callbacks.append(LearningRateScheduler(scheduler))

    # image augmentation
    # proper image augmentations are domain knowledge dependent.

    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-30, 30),
            cval=0
        ),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.ChangeColorTemperature((3500, 7000)),
        iaa.MultiplyHueAndSaturation((0.5, 1.5)),
        # iaa.CropAndPad(percent=(-0.05, 0.05))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=fit_params.get('epochs', 100),
                layers=fit_params.get('layers', 'heads'),
                custom_callbacks=callbacks,
                augmentation=augmentation)


def get_masks(image, mask, class_ids):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # print ("image shape: ", image.shape[1], image.shape[0])

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # cuong start
    if mask.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(mask.shape[-1]):
            semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            semantic_mask_one = semantic_mask_one * class_ids[i]
            semantic_masks = np.where(mask[:, :, i], semantic_mask_one, semantic_masks).astype(np.uint8)
            instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            instance_mask_one = instance_mask_one * (i + 1)
            instance_masks = np.where(mask[:, :, i], instance_mask_one, instance_masks).astype(np.uint8)

    return semantic_masks, instance_masks


def detect_and_get_masks(model, image):

    # Run model detection and generate the color splash effect
    # print("Running on {}".format(image_path))
    # Read image
    # image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    semantic_masks, instance_masks = get_masks(image, r['masks'], r['class_ids'])

    plt.subplot(1, 2, 1)
    plt.title('rgb')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('masks')
    plt.imshow(instance_masks)
    plt.show()

    # Save output
    file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, instance_masks)
    print("Saved to ", file_name)


def get_mask_rcnn_pretrained_file(dataset: str) -> str:

    assert dataset in ['coco', 'imagenet'], f"pretrained weight {dataset} is not available"

    dir_models = f"{get_project_dir()}/pre_trained_weights"

    if not os.path.isdir(dir_models):
        os.makedirs(dir_models)

    # Path to trained weights file
    path = f"{dir_models}/mask_rcnn_{dataset}.h5"

    return path


class InferenceConfig(FashionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_STRIDE = 1

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fashion objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/fashion/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--backbone', required=False,
                        metavar="feature extractor",
                        help="resnet50 or resnet101")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--object_classification', required=True,
                        metavar="label of annotation",
                        help='')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    if not os.path.isdir(args.logs):
        os.makedirs(args.logs)

    # Configurations
    if args.command == "train":
        config = FashionConfig(args.object_classification)
    else:
        class InferenceConfig(FashionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig(args.object_classification)

    if hasattr(args, 'backbone'):
        config.BACKBONE = args.backbone

    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = get_mask_rcnn_pretrained_file('coco')
        print(weights_path)
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes

        exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                   "mrcnn_bbox", "mrcnn_mask"]

        if model_architecture['exclude_rpn_model']:
            exclude.append('rpn_model')

        print(f"excluded layers: {exclude}")

        model.load_weights(weights_path, by_name=True, exclude=exclude)

    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.object_classification)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
