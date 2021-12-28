import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Concatenate, \
                                              Reshape
from tensorflow.compat.v1.keras import backend as K


class DecoderBlock(tf.keras.Model):

    def __init__(self, kernel_size, filter, block, data_format):

        super(DecoderBlock, self).__init__(name='')

        conv_name = f'decode_{block}_branch'

        self.conv2a = Conv2D(filter, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                             name=conv_name + '2a', data_format=data_format)
        self.conv2b = Conv2D(filter, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                             name=conv_name + '2b', data_format=data_format)

    def call(self, input_tensor, training=False):
        x = UpSampling2D()(input_tensor)
        x = self.conv2a(x)
        x = self.conv2b(x)

        return x


class EncoderBlock(tf.keras.Model):

    def __init__(self, kernel_size, filter, block, data_format):

        super(EncoderBlock, self).__init__(name='')

        conv_name = f'encode_{block}_branch'

        self.conv2a = Conv2D(filter, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                             name=conv_name + '2a', data_format=data_format)
        self.conv2b = Conv2D(filter, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                             name=conv_name + '2b', data_format=data_format)

    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.conv2b(x)
        x = MaxPooling2D()(x)

        return x

class Encoder(tf.keras.Model):

    def __init__(self, height, width, data_format, start_filters, latent_dim, conditioning_dim=0, name=''):

        super(Encoder, self).__init__(name=name)

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))

        self.start_filter = start_filters
        self.latent_dim = latent_dim
        self.height = height
        self.width = width

        def encode_block(filter, block):
            return EncoderBlock(3, filter, block, data_format=data_format)

        if conditioning_dim > 0:
            self.condition_layer = Dense(height*width, name='conditioning')

        self.l1a = encode_block(start_filters, block=1)
        self.l2a = encode_block(start_filters * 2, block=2)
        self.l3a = encode_block(start_filters * 4, block=3)
        self.l4a = encode_block(start_filters * 8, block=4)

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

        x = self.l1a(x)
        x = self.l2a(x)
        x = self.l3a(x)
        x = self.l4a(x)

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

        def decode_block(filter, block):
            return DecoderBlock(3, filter, block, data_format=data_format)

        self.embedding = Dense(np.prod(shape_spatial), name='embedding')

        start_filters = shape_spatial[-1]

        self.m1a = decode_block(start_filters, block=1)
        self.m2a = decode_block(start_filters//2, block=2)
        self.m3a = decode_block(start_filters//4, block=3)
        self.m4a = decode_block(start_filters//8, block=4)

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
        x = self.m4a(x)
        x = self.activation(x)

        return x


class VAEBasic(tf.keras.Model):

    def __init__(self, height, width, latent_dim, conditioning_dim=0,
                 start_filters=8, name='VAE'):

        super(VAEBasic, self).__init__(name=name)

        self.encoder = Encoder(height, width, data_format='channels_last', start_filters=start_filters,
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