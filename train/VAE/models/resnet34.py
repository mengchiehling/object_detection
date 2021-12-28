import string

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Concatenate, \
                                    Reshape, LayerNormalization, Conv2DTranspose
from tensorflow.keras import backend as K


class EncoderIdentityBlock(tf.keras.Model):
    """

    _IdentityBlock is the block that has no conv layer at shortcut.

    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or 'channels_last').
    """

    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(EncoderIdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(
            filters1, kernel_size, name=conv_name_base + '2a', data_format=data_format, padding='same')

        self.conv2b = Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)

        x += input_tensor
        return tf.nn.relu(x)


class EncoderConvBlock(tf.keras.Model):
    """_ConvBlock is the block that has a conv layer at shortcut.

    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
        'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
        conv layer at main path is with strides=(2,2), and the shortcut should
        have strides=(2,2) as well.
    """

    def __init__(self, kernel_size, filters, stage, block, data_format, strides=(2, 2)):
        super(EncoderConvBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(filters1, kernel_size, strides=strides, name=conv_name_base + '2a',
                             data_format=data_format, padding='same')

        self.conv2b = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
                             data_format=data_format)

        self.conv_shortcut = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '1',
                                    data_format=data_format, padding='same')

    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)

        shortcut = self.conv_shortcut(input_tensor)

        x += shortcut
        return tf.nn.relu(x)


class DecoderIdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, data_format):

        super(DecoderIdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = f'decoder_{stage}_{block}_branch'

        self.conv2a = Conv2D(
            filters1, kernel_size, name=conv_name_base + '2a', data_format=data_format)

        self.conv2b = Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)

        x += input_tensor
        return tf.nn.relu(x)

        return x


class DecoderConvBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, data_format, strides=(2, 2)):

        super(DecoderIdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = f'decode_conv_{stage}_{block}_branch'
        bn_name_base = f'decode_bn_{stage}_{block}_branch'

        self.conv2a = Conv2D(filters1, kernel_size, name=conv_name_base + '2a', data_format=data_format,
                             padding='same')

        self.conv2b = Conv2DTranspose(filters2, kernel_size, strides=strides, padding='same', name=conv_name_base + '2b',
                                      data_format=data_format)

        self.conv_shortcut = Conv2DTranspose(filters1, (1, 1), strides=strides, name=conv_name_base + '1',
                                             data_format=data_format)


        # self.lrn2a = LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True,
        #                               beta_initializer='zeros', gamma_initializer='ones',
        #                               beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
        #                               gamma_constraint=None, name=bn_name_base+'2a')


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)

        shortcut = self.conv_shortcut(input_tensor)

        x += shortcut
        return tf.nn.relu(x)


class Encoder(tf.keras.Model):

    def __init__(self, height, width, data_format, latent_dim, conditioning_dim=0, name=''):

        super(Encoder, self).__init__(name=name)

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))

        self.latent_dim = latent_dim
        self.height = height
        self.width = width

        if conditioning_dim > 0:
            self.condition_layer = Dense(height*width, name='conditioning')

        self.conv1 = Conv2D(
            64, (7, 7),
            strides=(2, 2),
            data_format=data_format,
            padding='same',
            name='conv1')

        self.max_pool = MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format, name='pool1', padding='same')

        structure_representation = [(2, 3), (3, 4), (4, 6), (5, 3)]  # (stage, number_of_blocks)

        self.custom_layers = []

        for s in structure_representation:
            stage, number_of_blocks = s
            filters = int(np.power(2, 4 + stage))
            if stage == 2:
                for idx, alphabet in enumerate(string.ascii_lowercase[:number_of_blocks]):
                    self.custom_layers.append(EncoderIdentityBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                   data_format=data_format))
            else:
                for idx, alphabet in enumerate(string.ascii_lowercase[:number_of_blocks]):
                    if idx == 0:
                        self.custom_layers.append(EncoderConvBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                   data_format=data_format))
                    else:
                        self.custom_layers.append(EncoderIdentityBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                       data_format=data_format))

        self.flatten = Flatten()
        self.mean = Dense(self.latent_dim, name='mean')
        self.noise = Dense(self.latent_dim, name='noise')

    def sampling(self, args):

        z_mean, z_log_sigma = args
        batch = tf.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch, self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def build_call(self, input_tensor, conditions=None, training=True):

        if conditions is not None:
            condition_up = self.condition_layer(conditions)
            condition_up = Reshape([self.height, self.width, 1])(condition_up)
            x = Concatenate(axis=3)([input_tensor, condition_up])
        else:
            x = input_tensor

        x = self.conv1(x)
        x = self.max_pool(x)
        for layer in self.custom_layers:
            x = layer(x)

        # for (224 * 224) inputs -> (7 * 7* 512)
        _, *shape_spatial = x.get_shape().as_list()

        x = self.flatten(x)

        z_mean = self.mean(x)
        z_log_sigma = self.noise(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_sigma])

        # kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

        return z, z_mean, z_log_sigma, shape_spatial


    def call(self, input_tensor, conditions=None, training=True):

        return self.build_call(input_tensor, conditions=conditions, training=training)


class Decoder(tf.keras.Model):

    def __init__(self, shape_spatial, data_format, name=''):

        super(Decoder, self).__init__(name=name)

        self.shape_spatial = shape_spatial

        self.embedding = Dense(np.prod(shape_spatial), name='embedding')

        structure_representation = [(2, 3), (3, 4), (4, 6), (5, 3)]  # (stage, number_of_blocks)

        self.custom_layers = []

        for s in structure_representation[::-1]:
            stage, number_of_blocks = s
            filters = int(np.power(2, 4 + stage))
            if stage == 2:
                for idx, alphabet in enumerate(string.ascii_lowercase[:number_of_blocks]):
                    self.custom_layers.append(DecoderIdentityBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                   data_format=data_format))
            else:
                for idx, alphabet in enumerate(string.ascii_lowercase[:number_of_blocks]):
                    if idx == (number_of_blocks-1):
                        self.custom_layers.append(DecoderConvBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                   data_format=data_format))
                    else:
                        self.custom_layers.append(DecoderIdentityBlock(3, [filters, filters], stage=stage, block=alphabet,
                                                                       data_format=data_format))

        self.activation = Conv2D(3, 1, activation='sigmoid')

    def call(self, input_tensor, conditions=None, training=True):

        if conditions is not None:
            x = Concatenate()([input_tensor, conditions])
        else:
            x = input_tensor

        x = self.embedding(x)
        x = Reshape(self.shape_spatial)(x)
        x = self.m1a(x)
        x = self.m2a(x)
        x = self.m3a(x)
        x = self.activation(x)

        return x

class VAEResnet34(tf.keras.Model):

    def __init__(self, height, width, latent_dim, conditioning_dim=0, name='VAE'):

        super(VAEResnet34, self).__init__(name=name)

        self.encoder = Encoder(height, width, data_format='channels_last',
                               latent_dim=latent_dim, conditioning_dim=conditioning_dim, name='encoder')

        trivial_images = tf.random.uniform((1, height, width, 3), maxval=1, dtype=tf.float32)
        if conditioning_dim != 0:
            trivial_conditions = tf.zeros((1, conditioning_dim), dtype=tf.float32)
        else:
            trivial_conditions = None
        _, z_mean, z_log_sigma, shape_spatial = self.encoder(trivial_images, trivial_conditions, training=False)

        self.decoder = Decoder(shape_spatial, data_format='channels_last', name='decoder')

    @property
    def trainable_weights(self):
        return (self.encoder.trainable_weights + self.decoder.trainable_weights)

    def call(self, input_tensor, conditions=None, training=True):

        z, z_mean, z_log_sigma, _ = self.encoder.build_call(input_tensor, conditions, training=training)
        output_tensor = self.decoder(z, conditions, training=training)

        return output_tensor, z_mean, z_log_sigma