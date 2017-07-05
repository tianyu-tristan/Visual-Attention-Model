from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Multiply
# from recurrentshop import RecurrentSequential

# from utils import weight_variable, bias_variable

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.legacy_seq2seq


class GlimpseNet(object):
    """Glimpse network.

    Take glimpse location input and output features for RNN.

    """

    def __init__(self, config, images_ph):
        self.original_size = config.original_size
        self.num_channels = config.num_channels
        self.sensor_size = config.sensor_size
        self.win_size = config.win_size
        self.minRadius = config.minRadius
        self.depth = config.depth
        self.scale = config.scale

        self.hg_size = config.hg_size
        self.hl_size = config.hl_size
        self.g_size = config.g_size
        self.loc_dim = config.loc_dim

        self.images_ph = images_ph

        self.init_layers()

    def init_layers(self):
        with tf.variable_scope("GlimpseNet", reuse=None):
            # Gimage
            self.conv1 = Conv2D(filters=64, kernel_size=(5,5), name="conv1")
            self.conv2 = Conv2D(filters=64, kernel_size=(3,3), name="conv2")
            self.conv3 = Conv2D(filters=128, kernel_size=(3,3), name="conv3")
            self.flatten = Flatten(name="flatten")
            self.dense = Dense(units=1024, activation='relu', name="img_dense")
            # Gloc
            self.loc_dense = Dense(units=1024, activation='relu', name="loc_dense")
            # Aux
            self.multiply = Multiply(name="multiply")

    def get_glimpse(self, loc):
        """Take glimpse on the original images."""
        imgs = tf.reshape(self.images_ph, [
                tf.shape(self.images_ph)[0], self.original_size, self.original_size,
                self.num_channels
        ])
        scaled_glimpse_imgs = []
        for i in range(1, self.scale+1):
            glimpse_imgs = tf.image.extract_glimpse(input=imgs,
                                                    size=[self.win_size * i, self.win_size * i],
                                                    offsets=loc)
            glimpse_imgs = tf.image.resize_bilinear(images=glimpse_imgs,
                                                    size=[self.win_size, self.win_size])

            scaled_glimpse_imgs.append(glimpse_imgs)

        return tf.concat(scaled_glimpse_imgs, axis=1) # (batch, win_size, win_size * scale, num_channels)

    def __call__(self, loc):
        with tf.variable_scope("GlimpseNet", reuse=True):
            # Gimage
            glimpse_imgs = self.get_glimpse(loc)

            conv = self.conv1(glimpse_imgs)
            conv = self.conv2(conv)
            conv = self.conv3(conv)
            flatten = self.flatten(conv)
            Gimage = self.dense(flatten)

            # Gloc
            Gloc = self.loc_dense(loc)

            # Eq(1)
            g = self.multiply([Gimage, Gloc]) # (batch, timesteps, 1024)

        return g


class EmissionNet(object):
    """Location network.

    Take output from other network and produce and sample the next location.

    """

    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_size
        self.loc_std = config.loc_std
        self._sampling = True

        self.init_weights()

    def init_weights(self):
        with tf.variable_scope("EmissionNet"):
            self.w = tf.get_variable(name="W",
                                     shape=[self.input_dim, self.loc_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b",
                                     shape=[self.loc_dim,],
                                     initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, input):
        mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
        mean = tf.stop_gradient(mean)
        if self._sampling:
            loc = mean + tf.random_normal(
                    (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean
        # E = Sequential()
        # E.add(Flatten())
        # E.add(Dense(1024))
        # E.add(Activation('relu'))
        # E.add(Dense(self.loc_dim))
        #
        # l = E(input)
        # return l

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        self._sampling = sampling


class BaselineNet(object):
    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_size
        self.loc_std = config.loc_std
        self._sampling = True

        self.init_weights()

    def init_weights(self):
        with tf.variable_scope("BaselineNet"):
            self.w = tf.get_variable(name="W",
                                     shape=[self.input_dim, 1],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b",
                                     shape=[1, ],
                                     initializer=tf.contrib.layers.xavier_initializer())
    def __call__(self, input):
        baseline_t = tf.nn.xw_plus_b(input, self.w, self.b)
        baseline_t = tf.squeeze(baseline_t) # (1)

        return baseline_t


class ClassifyNet(object):
    """Classification network

    Take output from previous glimpse network and predict class label
    """
    def __init__(self, config):
        self.num_classes = config.num_classes
        with tf.variable_scope("ClassifyNet", reuse=None):
            self.dense = Dense(units=self.num_classes, activation='softmax')

    def __call__(self, input):
        # r1_output = Flatten()(input)
        with tf.variable_scope("ClassifyNet", reuse=True):
            P = self.dense(input)
        return P

class ContextNet(object):
    """Context network

    Maps down-sampled low-resolution version of input image, to a feature vector which is used to initialize RNN
    """
    def __init__(self, config):
        self.cell_size = config.cell_size
        with tf.variable_scope("ContextNet", reuse=None):
            self.conv1 = Conv2D(filters=64, kernel_size=(5, 5), name="conv1")
            self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), name="conv2")
            self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), name="conv3")
            self.flatten = Flatten(name="flatten")
            self.dense = Dense(units=512, activation='relu', name="dense")

    def __call__(self, imgs):
        with tf.variable_scope("ContextNet", reuse=True):
            conv = self.conv1(imgs)
            conv = self.conv2(conv)
            conv = self.conv3(conv)
            flatten = self.flatten(conv)
            context = self.dense(flatten)

        return context
