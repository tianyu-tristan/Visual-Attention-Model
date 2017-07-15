import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import sys
from GlimpseSensor import *
from Globals import *
from CoreRNN import *

batchSize = constants['batchSize']
trainingIters = 1000000  # in terms of sample size
displayStep = 1  # how often to print details
step = 0

logPath = "logs"
modelPath = "model/"


def setupSummaries():
    with tf.variable_scope('monitor') as scope:
        loss = tf.Variable(0.0)
        tf.summary.scalar("Loss", loss)
        trainAcc = tf.Variable(0.0)
        tf.summary.scalar("Train Accuracy", trainAcc)
        testAcc = tf.Variable(0.0)
        tf.summary.scalar("Test Accuracy", testAcc)
        summaryVars = [loss, trainAcc, testAcc]
        summaryPlaceholders = [
            tf.placeholder("float") for i in range(
                len(summaryVars))]
        updateOps = [
            summaryVars[i].assign(
                summaryPlaceholders[i]) for i in range(
                len(summaryVars))]
        return summaryPlaceholders, updateOps

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
monitorPh, monitorOps = setupSummaries()
saver = tf.train.Saver()
summaryOps = tf.summary.merge_all()
glimpseSensor = GlimpseSensor()
coreRnn = CoreRNN(glimpseSensor)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.summary.FileWriter(logPath, sess.graph)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    checkpoint = tf.train.get_checkpoint_state(modelPath)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("successfully loaded checkpoint")

    while step * batchSize < trainingIters:
        batch = mnist.train.next_batch(batchSize)
        coreRnn.forward(batch[0], batch[1])
        print ("finished forward...")
        sess.run([coreRnn.optimizer], feed_dict={
                 coreRnn.imgsPlaceholder: batch[0], coreRnn.labelsPlaceholder: batch[1]})
        print ("running optimizer...")
        if step % displayStep == 0:
            # Calculate training loss and accuracy
            loss, trainAcc = sess.run([coreRnn.cost, coreRnn.accuracy], feed_dict={
                coreRnn.imgsPlaceholder: batch[0],
                coreRnn.labelsPlaceholder: batch[1]})
            # calculate test accuracy
            tbatch = mnist.test.next_batch(batchSize)
            coreRnn.forward(tbatch[0], tbatch[1], sess)
            testAcc = sess.run(
                coreRnn.accuracy,
                feed_dict={
                    coreRnn.imgsPlaceholder: batch[0],
                    coreRnn.labelsPlaceholder: batch[1]})
            sess.run([monitorOps[0], monitorOps[1], monitorOps[2]], feed_dict={
                     monitorPh[0]: float(loss), monitorPh[1]: trainAcc, monitorPh[2]: testAcc})

            print("Iter " +
                  str(step *
                      batchSize) +
                  ", Minibatch Loss= " +
                  "{:.6f}".format(loss) +
                  ", Training Accuracy= " +
                  "{:.5f}".format(trainAcc) +
                  ", Test Accuracy= " +
                  "{:.5f}".format(testAcc))

            savePath = saver.save(sess, modelPath + "dram.ckpt")
            print("Model saved in file: %s" % savePath)
            summaryStr = sess.run(
                summaryOps,
                feed_dict={
                    coreRnn.imgsPlaceholder: batch[0],
                    coreRnn.labelsPlaceholder: batch[1]})
            writer.add_summary(summaryStr, step)
        step += 1
