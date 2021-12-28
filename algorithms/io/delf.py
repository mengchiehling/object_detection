import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from algorithms.settings import delf_architecture

dir_model = '' # path to where the model will be saved


def load():

    config = tf.ConfigProto(inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0)

    graph = tf.Graph()

    with graph.as_default():
        link = delf_architecture['LINK']
        delf_module = hub.Module(link)
        image_placeholder = tf.compat.v1.placeholder(tf.compat.v1.float32,
                                                     shape=(None, None, 3), name='input_image')
        module_inputs = {'image': image_placeholder}
        module_inputs.update(delf_architecture['MODULE_INPUTS'])
        module_outputs = delf_module(module_inputs, as_dict=True)

        init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    session = tf.compat.v1.Session(graph=graph, config=config)
    session.run(init_op)

    image = np.zeros(shape=(224, 224, 3)).astype(np.uint8)

    with graph.as_default():
        session.run([module_outputs['locations'],
                     module_outputs['descriptors']],
                     feed_dict={image_placeholder: image/255})

        tensor_info_image = tf.compat.v1.saved_model.utils.build_tensor_info(image_placeholder)
        tensor_info_locations = tf.compat.v1.saved_model.utils.build_tensor_info(module_outputs['locations'])
        tensor_info_descriptors = tf.compat.v1.saved_model.utils.build_tensor_info(module_outputs['descriptors'])
        signature = tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={'image': tensor_info_image},
                        outputs={'locations': tensor_info_locations,
                                 'descriptors': tensor_info_descriptors},
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                    )

        export_dir = os.path.join(dir_model, 'delf_v1')
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

        # legacy_init_op = tf.group(tf.global_variables_initializer(
        #         ), tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
            # legacy_init_op=legacy_init_op
        )

        builder.save()
        print('Done exporting!')


class DELFWrapper():

    def __init__(self):

        """

        :param train: in the sense of tensorflow.train, which can be considered as a way of high effiency data processing
        """

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        # load the protobuf file

        export_dir = os.path.join(dir_model, 'delf_v1')
        meta_graph_def = tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], export_dir)

        signature_def = meta_graph_def.signature_def
        signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # Get input tensor
        self.input_image_tensor = signature.inputs['image'].name

        # Get output tensor
        self.locations_tensor = signature.outputs['locations'].name
        self.descriptors_tensor = signature.outputs['descriptors'].name

    def predict(self, img: np.ndarray):

        """

        :param img: a uint8 nd.ndarray with dimension H, W, 3
        :return:
        """

        assert img.shape[2] == 3, 'the image does not have the right color channels'

        with self.graph.as_default():
            with self.session.as_default() as sess:
                image_local_feature = sess.run([self.locations_tensor, self.descriptors_tensor],
                                                 {self.input_image_tensor: img / 255})

        return image_local_feature