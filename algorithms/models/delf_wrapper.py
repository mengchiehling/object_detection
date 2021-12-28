'''
Although the likelyhood that this image feature extractor will be depreciated is very high,
it is kept for the moment

Example
from PIL import Image
import numpy as np

from visual_search.io.path_definition import get_project_dir

image = Image.open(f'{get_project_dir()}/adidas_test_image.jpg')

image_np = np.array(image)

delf_wrapper = DELFWrapper()

delf_wrapper.predict(image_np)

'''
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class DELFWrapper:

    '''
    https://tfhub.dev/google/delf/1
    '''

    def __init__(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.delf_module = hub.Module("https://tfhub.dev/google/delf/1")
            self.image_placeholder = tf.compat.v1.placeholder(tf.compat.v1.float32,
                                                              shape=(None, None, 3), name='input_image')
            module_inputs = {'image': self.image_placeholder,
                             'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
                             'score_threshold': 100,
                             'max_feature_num': 1000}

            self.module_outputs = self.delf_module(module_inputs, as_dict=True)

            init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

        config = tf.ConfigProto(inter_op_parallelism_threads=0,
                                intra_op_parallelism_threads=0)

        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        self.session.run(init_op)

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        with self.graph.as_default():
            if isinstance(image, str):
                image = tf.image.decode_jpeg(image, channels=3)

            float_image = image/255

            locations, descriptors = self.session.run([self.module_outputs['locations'],
                                                       self.module_outputs['descriptors']],
                                                       feed_dict={self.image_placeholder: float_image})

        return locations, descriptors
