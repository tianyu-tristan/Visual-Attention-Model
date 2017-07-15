import tensorflow as tf
import numpy as np
from Globals import *
from BaseNet import *


class LocationNet(BaseNet):

    def __init__(self):
        self.meanList = []
        self.sampleList = []
        hiddenDim = constants['hiddenDim']
        self.std = constants['locationStd']
        with tf.variable_scope('locationnet') as scope:
            self.w = self.variableWithWeightDecay(
                'weight', [hiddenDim, 2], 1e-4, 0.0)
            self.b = self.variableOnCpu(
                'bias', [2], tf.constant_initializer(0.0))

    def clear(self):
        self.meanList = []
        self.sampleList = []

    def forward(self, hiddenState):
        unitPixels = constants['unitPixels']
        imageSize = constants['imageSize']
        means = tf.nn.bias_add(tf.matmul(hiddenState, self.w), self.b)
        self.boundedMeans = tf.tanh(means)
        self.meanList.append(self.boundedMeans)
        dist = tf.contrib.distributions.Normal(
            self.boundedMeans, self.std)
        self.samples = dist.sample(1)
        self.samples = self.samples[0, :, :]
        self.sampleList.append(self.samples)
        boundedSamples = tf.tanh(self.samples)
        ratio = float(unitPixels) / float(imageSize)
        self.locations = tf.multiply(boundedSamples, ratio)
        return self.locations

    def backward(self, baseline, rewards):
        alpha = constants['alpha']
        params = [self.w, self.b]
        policyGradients = []
        for i in range(len(self.meanList)):
            dlog = (self.sampleList[i] - self.meanList[i]) / (self.std * self.std)
            advantages = rewards - baseline
            advantages = tf.expand_dims(advantages, 1)
            dpolicy = alpha * advantages * dlog
            policyGradients.append(tf.gradients(self.meanList[i], params, dpolicy))
        optimize = tf.train.AdamOptimizer(0.001)
        gradsAndVarsAllSteps = [zip(g, params) for g in policyGradients]
        for gradsAndVars in gradsAndVarsAllSteps:
            locationNetTrainOp = optimize.apply_gradients(gradsAndVars)
        return locationNetTrainOp
