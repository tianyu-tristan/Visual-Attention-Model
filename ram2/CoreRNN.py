import tensorflow as tf
import numpy as np
from Globals import *
from LocationNet import *
from GlimpseNet import *
from BaseNet import *


class CoreRNN(BaseNet):

    def __init__(self, glimpseSensor):
        numClasses = constants['numClasses']
        imageSize = constants['imageSize']
        self.imgsPlaceholder = tf.placeholder(
            tf.float32, [None, imageSize * imageSize])
        self.labelsPlaceholder = tf.placeholder(tf.float32, [None, numClasses])
        self.numSteps = constants['numGlimpses']
        self.batchSize = constants['batchSize']
        self.hiddenDim = constants['hiddenDim']
        self.numClasses = constants['numClasses']
        self.learningRate = constants['learningRate']
        self.locationNetwork = LocationNet()
        self.glimpseNetwork = GlimpseNet()
        self.glimpseSensor = glimpseSensor
        self.adamOptimizer = tf.train.AdamOptimizer(
            learning_rate=self.learningRate)
        with tf.variable_scope('corernn') as scope:
            self.wx = self.variableWithWeightDecay(
                'weightsGlimpse', [256, 4 * self.hiddenDim], 1e-4, 0.0)
            self.wh = self.variableWithWeightDecay(
                'weightsPrevh', [256, 4 * self.hiddenDim], 1e-4, 0.0)
            self.b = self.variableOnCpu(
                'biasCoreRnn', [self.hiddenDim * 4], tf.constant_initializer(0.0))

        with tf.variable_scope('classifier') as scope:
            self.wc = self.variableWithWeightDecay(
                'weightClassifier', [
                    self.hiddenDim, self.numClasses], 1e-4, 0.0)
            self.bc = self.variableOnCpu(
                'biasClassifier', [
                    self.numClasses], tf.constant_initializer(0.0))

    def forward(self, imgs, labels):
        hprev = [[0.0] * self.hiddenDim for i in range(self.batchSize)]
        cprev = 0.0
        self.locationNetwork.clear()
        for t in range(self.numSteps):
            hprev, cprev = self.stepForward(hprev, cprev, imgs)

        logits = tf.nn.bias_add(tf.matmul(hprev, self.wc), self.bc)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, self.labelsPlaceholder, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy_average')
        tf.add_to_collection('losses', cross_entropy_mean)

        self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.optimizer = self.adamOptimizer.minimize(self.cost)
        correctPred = tf.equal(
            tf.argmax(
                logits, 1), tf.argmax(
                self.labelsPlaceholder, 1))
        correctPred = tf.cast(correctPred, tf.float32)
        self.accuracy = tf.reduce_mean(correctPred)

        # calculate rewards and baselines
        baseline = self.accuracy
        rewards = correctPred

        # train location network
        self.locationNetTrainOp = self.locationNetwork.backward(baseline, rewards)

    def stepForward(self, preH, preC, imgs):
        nextH, nextC = None, None

        locations = self.locationNetwork.forward(preH)
        glimpses = self.glimpseSensor.extract(imgs, self.imgsPlaceholder, locations)
        glimpseEncodings = self.glimpseNetwork.forward(glimpses, locations)

        H = 256
        activationVector = tf.nn.bias_add(
            tf.add(
                tf.matmul(
                    glimpseEncodings, self.wx), tf.matmul(
                    preH, self.wh)), self.b)
        ai = activationVector[:, 0:H]
        af = activationVector[:, H:2 * H]
        ao = activationVector[:, H * 2:3 * H]
        ag = activationVector[:, H * 3:4 * H]

        i = tf.sigmoid(ai)
        f = tf.sigmoid(af)
        o = tf.sigmoid(ao)
        g = tf.tanh(ag)

        nextC = tf.add(tf.multiply(preC, f), tf.multiply(i, g))
        tanhCell = tf.tanh(nextC)
        nextH = tf.multiply(o, tanhCell)

        return nextH, nextC
