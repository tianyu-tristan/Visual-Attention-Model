"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

try:
  xrange
except NameError:
  xrange=range

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.legacy_seq2seq

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
data = np.load('../data/mnist_digit_sample_8dsistortions9x9.npz')

# the data, shuffled and split between train and test sets
# x_train = np.expand_dims(data['X_train'], axis=-1)
# y_train = to_categorical(np.reshape(data['y_train'], (-1)))
# x_va = np.expand_dims(data['X_valid'], axis=-1)
# y_va = to_categorical(np.reshape(data['y_valid'], (-1)))
# x_test = np.expand_dims(data['X_test'], axis=-1)
# y_test = to_categorical(np.reshape(data['y_test'], (-1)))
x_train = np.reshape(data['X_train'], (-1, 10000))
y_train = np.reshape(data['y_train'], (-1))
x_va = np.reshape(data['X_valid'], (-1, 10000))
y_va = np.reshape(data['y_valid'], (-1))
x_test = np.reshape(data['X_test'], (-1, 10000))
y_test = np.reshape(data['y_test'], (-1))


x_train = x_train.astype('float32')
x_va = x_va.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_va.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

config = Config()

input_shape = (config.original_size, config.original_size, 1)

n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = x_train.shape[0] // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for i in xrange(n_steps):
    steps_per_epoch = x_train.shape[0] // config.batch_size
    num_samples = steps_per_epoch * config.batch_size
    for step in range(steps_per_epoch -1):
      start = step * config.batch_size
      end = (step+1) * config.batch_size
      images, labels = x_train[start:end], y_train[start:end]
      # duplicate M times, see Eqn (2)
      images = np.tile(images, [config.M, 1])
      labels = np.tile(labels, [config.M])
      loc_net.samping = True
      adv_val, baselines_mse_val, xent_val, logllratio_val, \
          reward_val, loss_val, lr_val, _ = sess.run(
              [advs, baselines_mse, xent, logllratio,
               reward, loss, learning_rate, train_op],
              feed_dict={
                  images_ph: images,
                  labels_ph: labels
              })
    if i and i % 100 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))

    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [(x_va, y_va), (x_test, y_test)]:
        steps_per_epoch = dataset[0].shape[0] // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.eval_batch_size
        loc_net.sampling = True
        for test_step in xrange(steps_per_epoch-1):
          images, labels = dataset[0][test_step * config.batch_size : (test_step+1) * config.batch_size]
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1])
          labels = np.tile(labels, [config.M])
          softmax_val = sess.run(softmax,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / num_samples
        if dataset[0] == x_va:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))
