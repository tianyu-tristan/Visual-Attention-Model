import tensorflow as tf
from Globals import *
from BaseNet import *


class GlimpseNet(BaseNet):

    def __init__(self):
        self.imageSize = constants['imageSize']
        self.imageChannel = constants['imageChannel']
        self.numGlimpseResolution = constants['numGlimpseResolution']
        self.glimpseOutputSize = constants['glimpseOutputSize']
        self.glimpseDim = self.imageChannel * self.glimpseOutputSize * \
            self.glimpseOutputSize * self.numGlimpseResolution

        # linear layer processing retina encoding
        with tf.variable_scope('g0') as scope:
            self.wg0 = self.variableWithWeightDecay(
                'weights', [self.glimpseDim, 128], 1e-4, 0.0)
            self.bg0 = self.variableOnGpu(
                'biases', [128], tf.constant_initializer(0.0))

        # linear layer processing location
        with tf.variable_scope('g1') as scope:
            self.wg1 = self.variableWithWeightDecay(
                'weights', [2, 128], 1e-4, 0.0)
            self.bg1 = self.variableOnGpu(
                'biases', [128], tf.constant_initializer(0.0))

        # linear layer processing previouse two linear layers
        with tf.variable_scope('g2') as scope:
            self.wg2 = self.variableWithWeightDecay(
                'weights', [256, 256], 1e-4, 0.0)
            self.bg2 = self.variableOnGpu(
                'biases', [256], tf.constant_initializer(0.0))

    def forward(self, glimpses, locations):
        glimpses = tf.reshape(glimpses, [-1, self.glimpseDim])
        out = tf.matmul(glimpses, self.wg0)
        bias = tf.nn.bias_add(out, self.bg0)
        self.g0 = tf.nn.relu(bias)
        # self.activationSummary(self.g0)

        out = tf.matmul(locations, self.wg1)
        bias = tf.nn.bias_add(out, self.bg1)
        self.g1 = tf.nn.relu(bias)
        # self.activationSummary(self.g1)

        combined = tf.concat([self.g0, self.g1], axis=1)
        out = tf.matmul(combined, self.wg2)
        bias = tf.nn.bias_add(out, self.bg2)
        self.g2 = tf.nn.relu(bias)
        # self.activationSummary(self.g2)

        return self.g2
