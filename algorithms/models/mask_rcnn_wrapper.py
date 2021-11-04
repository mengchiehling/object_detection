'''
Usage:

instance initialization:

wrapper = MaskRcnnInferenceWrapper()

load a picture as a np.ndarray

predict:

rois, masks, scores, class_ids = wrapper.predict(<img as np.ndarray>)

'''


import os

import numpy as np
import tensorflow as tf

import train.Mask_RCNN.mrcnn.model as modellib
from train.Mask_RCNN.fashion.fashion import InferenceConfig
from algorithms.utils.utils_basic import load_model_key
from algorithms.io.path_definition import get_project_dir

mask_rcnn_key = load_model_key('Mask_RCNN')


class MaskRcnnInferenceWrapper:

    '''
    Usage:

    wrapper = MaskRcnnInferenceWrapper()

    img = cv2.imread(full_filename)
    img = img[:, :, [2, 1, 0]]

    mask_rcnn_wrapper.predict(img)

    Make sure the input image is in RGB format

    class_id:
        0: shoe
        1: kleider
        2: t-shirt
    '''

    id_category_mapping = {0: 'shoe',
                           1: 'kleider',
                           2: 't-shirt'}

    def __init__(self):
        """
        initialize mask rcnn network with in a graph and a session within this instance
        download the model weight from aws if it is not available in local/server hard drive
        model version is specific in config/app_config,
        INFERENCE_MODEL:
            RCNN:
                Mask_RCNN:
        It is aws location is c24-fa-ds-object-detection/trained_models/object_detection/RCNN

        """
        config = InferenceConfig()
        config.display()

        # attach a graph and a session to the instance such that the model runs in multithreading program
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.mask_rcnn_model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

        config = tf.ConfigProto(inter_op_parallelism_threads=0,
                                intra_op_parallelism_threads=0)

        dir_model = os.path.join(get_project_dir(), 'data', 'app', 'models')
        _weights = os.path.join(dir_model, mask_rcnn_key, "best_model.h5")

        assert os.path.isfile(_weights), f"{_weights} does not exist"

        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with self.sess.as_default():
                self.mask_rcnn_model.load_weights(_weights, by_name=True)

    def predict(self, img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, int):
        """

        Args;
            img: a uint8 nd.ndarray with dimension H, W, 3
        Returns:
        """

        assert img.shape[2] == 3, 'the image does not have the right color channels'

        with self.graph.as_default():
            with self.sess.as_default():
                results = self.mask_rcnn_model.detect([img], verbose=1)

        r = results[0]
        rois = r['rois']
        masks = r['masks']
        scores = r['scores']
        class_ids = r['class_ids']

        return rois, masks, scores, class_ids
