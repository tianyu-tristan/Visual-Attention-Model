from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from networks import GlimpseNet, EmissionNet, ContextNet, ClassifyNet, BaselineNet
from config import Config


rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.legacy_seq2seq
distributions = tf.contrib.distributions
logging.basicConfig(filename='dram.log',level=logging.DEBUG)

import pickle

def truncate_labels(labels):
    """
    (1) replacing row[0] by 10, and move it to the last of row
    (2) replace the second 10 by -1 row wise
    """
    def do_one_row(row):
        erase = False
        for i, _ in enumerate(row):
            if erase:
                row[i] = -1
            else:
                if row[i] == 10:
                    erase = True
        return row

    ret = np.copy(labels)
    ret = repair_labels(ret)
    return np.apply_along_axis(do_one_row, axis=1, arr=ret)

def repair_labels(labels):
    """
    replacing row[0] by 10, and move it to the last of row
    :param labels:
    :return:
    """
    ret = np.copy(labels)
    ret[:, 0] = 10  # overwrite length to be stop seq
    ret = np.roll(ret, -1, axis=1)  # move first to last
    return ret

def mask_labels(labels):
    """
    (1) replacing row[0] by 10, and move it to the last of row
    (2) replace the second 10 by -1 row wise
    """
    def do_one_row(row):
        erase = False
        for i, _ in enumerate(row):
            if erase:
                row[i] = 0
            else:
                if row[i] == 10:
                    erase = True
                row[i] = 1
        return row

    ret = np.copy(labels)
    return np.apply_along_axis(do_one_row, axis=1, arr=ret)

print('Loading pickled data...')

pickle_file = 'SVHN.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_train = save['train_dataset']
    Y_train = save['train_labels']
    Y_train = repair_labels(Y_train)
    Y_train_mask = mask_labels(Y_train)
    X_val = save['valid_dataset']
    Y_val = save['valid_labels']
    Y_val = repair_labels(Y_val)
    Y_val_mask = mask_labels(Y_val)
    X_test = save['test_dataset']
    Y_test = save['test_labels']
    Y_test = repair_labels(Y_test)
    Y_test_mask = mask_labels(Y_test)
    del save  
    print('Training data shape:', X_train.shape)
    print('Training label shape:',Y_train.shape)
    print('Validation data shape:', X_val.shape)
    print('Validation label shape:', Y_val.shape)
    print('Test data shape:', X_test.shape)
    print('Test label shape:', Y_test.shape)

print('Data successfully loaded!')

# TO BE REMOVED

X_train = X_train[:128]
Y_train = Y_train[:128]

# Load config & data

config = Config()

#X_train = np.random.randn(config.batch_size, config.original_size, config.original_size, config.num_channels)
#Y_train = np.random.randint(0, 2, (config.batch_size, config.max_num_digits+1, config.num_classes))

# placeholders
X_ph = tf.placeholder(dtype=tf.float32,
                      shape=[None, config.original_size, config.original_size, config.num_channels],
                      name="X")
# want y = [1,5,8,10,-1,-1], where 0-9 is classes, 10 is stop sign, -1 is padding
Y_ph = tf.placeholder(dtype=tf.int64,
                      shape=[None, config.max_num_digits+1],
                      name="Y")

Y_mask_ph = tf.placeholder(dtype=tf.int64,
                           shape=[None, config.max_num_digits+1],
                           name="Y_mask")

curriculum_ph = tf.placeholder(dtype=tf.float32,
                               shape=[None, config.max_num_digits+1],
                               name="curriculum")

rewards_ph = tf.placeholder(dtype=tf.float32,
                            shape=[None, config.max_num_digits+1],
                            name="rewards")

# with tf.variable_scope("DRAM", reuse=None):

# Emission Net
# with tf.variable_scope("EmissionNet", reuse=None):
Enet = EmissionNet(config=config)

# Baseline Net
# with tf.variable_scope("BaselineNet", reuse=None):
Bnet = BaselineNet(config=config)

# Glimpse Net
# with tf.variable_scope("GlimpseNet", reuse=None):
Gnet = GlimpseNet(config=config, images_ph=X_ph)

# Context Net
# with tf.variable_scope("ContextNet", reuse=None):
Cnet = ContextNet(config=config)

# Classification Net
# with tf.variable_scope("ClassifyNet", reuse=None):
Onet = ClassifyNet(config=config)

# Build Recurrent Net
timesteps = config.num_glimpses * (config.max_num_digits + 1) # 3 * (5 + 1) = 18


# functions
# start location
def get_next_input(r2_outputs, E, G):
    loc, mean = E(r2_outputs)
    # with tf.variable_scope("GlimpseNet", reuse=True):
    r1_inputs = G(loc) # (batch_size, 1024)
    loc_mean_arr.append(mean)
    sampled_loc_arr.append(loc)
    return r1_inputs


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)

  logll = gaussian.log_prob(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]

  return logll


def likelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)

  ll = gaussian.prob(sampled) #[timesteps, batch_sz, loc_dim]
  ll = tf.reduce_prod(ll, 2)
  ll = tf.transpose(ll) # [batch_sz, timesteps]

  return ll


def build_curriculum(scores):
    """use current scores of prediction to build curriculum mask

    :param scores: numpy ndarray of shape (batch_size, max_digits+1)
    :return: numpy ndarray of shape (batch_size, max_digits+1)
    """
    def do_one_row(row):
        failed_once = False
        for i, _ in enumerate(row):
            if failed_once:
                row[i] = 0
            else:
                if row[i] == 0:
                    row[i] = 1
                    failed_once = True
        return row
    tmp = np.copy(scores)
    curriculum_mask = np.apply_along_axis(do_one_row, axis=1, arr=tmp)
    return curriculum_mask

def build_rewards(scores):
    """use current scores of prediction to build rewards mask

    :param scores: numpy ndarray of shape (batch_size, max_digits+1)
    :return: numpy ndarray of shape (batch_size, max_digits+1)
    """
    def do_one_row(row):
        failed_once = False
        for i, _ in enumerate(row):
            if failed_once:
                row[i] = 0
            else:
                if row[i] == 0:
                    row[i] = 0
                    failed_once = True
        return row

    tmp = np.copy(scores)
    rewards_mask = np.apply_along_axis(do_one_row, axis=1, arr=tmp)
    return rewards_mask


# def curriculum(row):
#     with tf.variable_scope("curriculum", reuse=True):
#         failed_once = tf.Variable(False, dtype=tf.bool, name="failed_once")
#         correct_up_to = tf.Variable([0], name="correct_up_to")
#         def failed_once_alrdy():
#             return True
#         def never_failed_yet():
#             correct_up_to = tf.get_variable("correct_up_to")
#             tf.scatter_add(correct_up_to, [0],[1])
#             return True
#
#         def mislabel():
#             failed_once = tf.get_variable("failed_once")
#             tf.cond(failed_once, failed_once_alrdy, never_failed_yet)
#             failed_once.assign(True)
#             return True
#
#         def correctlabel():
#             correct_up_to = tf.get_variable("correct_up_to")
#             tf.scatter_add(correct_up_to, [0], [1])
#             return False
#
#         for element in tf.unstack(row):
#             tf.cond(tf.equal(element, 0), true_fn=mislabel, false_fn=correctlabel)
#
#         return correct_up_to


# Start Recurrent Network
# with tf.variable_scope("DRAM", reuse=None):
# start context
init_state = Cnet(X_ph)  # (batch_size, 512)

r1 = rnn_cell.LSTMCell(config.cell_size)
r2 = rnn_cell.LSTMCell(config.cell_size)

r1_zero_state = r1.zero_state(config.batch_size, tf.float32)
r1_zero_inputs = tf.zeros((config.batch_size, config.cell_size * config.scale), dtype=tf.float32)
r2_zero_inputs = tf.zeros((config.batch_size, config.cell_size), dtype=tf.float32)
r2_init_state = tf.nn.rnn_cell.LSTMStateTuple(init_state, init_state)  # init_state = (batch_size, cell_size)

with tf.variable_scope("r1", reuse=None):
    r1_outputs, r1_state = r1(inputs=r1_zero_inputs, state=r1_zero_state)
with tf.variable_scope("r2", reuse=None):
    r2_outputs, r2_state = r2(inputs=r2_zero_inputs, state=r2_init_state)

outputs = []
baselines = []
pred_labels = []
probs = []

loc_mean_arr = []
sampled_loc_arr = []

for t in range(timesteps):

    # with tf.variable_scope("DRAM", reuse=True):
    r1_inputs = get_next_input(r2_outputs, Enet, Gnet)

    with tf.variable_scope("r1", reuse=True):
        r1_outputs, r1_state = r1(inputs=r1_inputs, state=r1_state)
    with tf.variable_scope("r2", reuse=True):
        r2_outputs, r2_state = r2(inputs=r1_outputs, state=r2_state)


    P = Onet(r1_outputs) # (batch_size, num_classes)

    probs.append(P)
    pred_labels.append(tf.argmax(input=P, axis=1))
    outputs.append(r1_outputs)

    baseline_t = Bnet(r2_outputs) # (batch_size, 1)
    # logging.debug("baseline_t = {}".format(sess.run(baseline_t)))
    # logging.debug("Bnet.w = {}".format(Bnet.w))
    baselines.append(baseline_t)
        # # save the current glimpse and the hidden state
        # inputs[t] = glimpse
        # outputs[t] = hiddenState
        # # get the next input glimpse
        # if t != self.timesteps - 1:
        #     glimpse = get_next_input(hiddenState)
        # else:
        #     baseline = tf.sigmoid(tf.matmul(hiddenState, Wb_h_b) + Bb_h_b)
        #     baselines.append(baseline)


# Eq(15) MLE objective function
Y_true_onehot = tf.stack([Y_ph] * config.num_glimpses, axis=2) # (batch_sz, max_digits+1, num_glimpse)
Y_true_onehot = tf.one_hot(indices=Y_true_onehot,
                           depth=11, on_value=1.,
                           off_value=0.,
                           axis=3,
                           dtype=tf.float32) # (batch_sz, max_digits+1, num_glimpse, num_classes)
                                             # Y_true_onehot will have row of all 0s if the target label is -1 (padding)

probs = tf.stack(probs) # (timesteps=18, batch_size, num_classes)
probs = tf.reshape(probs, [config.max_num_digits+1, config.num_glimpses, config.batch_size, config.num_classes])
probs = tf.transpose(probs, perm=[2,0,1,3]) # probs = (batch_size, max_digits+1=6, num_glimpse=3, num_classes)

# Eq(14)
Y_pred = tf.reduce_mean(tf.log(probs), axis=2) # (batch_size, max_digits+1, num_classes)
Y_pred = tf.argmax(Y_pred, axis=2) # (batch_size, max_digits+1)
Y_true = Y_ph # (batch_size, max_digits+1)

scores = tf.cast(tf.equal(Y_pred, Y_true), dtype=tf.int64) # (batch_size, max_digits+1), where -1 padding is converted to 0

p_ys = tf.multiply(probs, Y_true_onehot) # mask out incorrect label probability, then reduce sum to get the correct prob
p_ys = tf.reduce_sum(p_ys, axis=3) # p_ys = [batch_sz, max_digits+1, num_glimpse)

p_ls = likelihood(loc_mean_arr, sampled_loc_arr, config.loc_std) # [batch_sz, timesteps]
p_ls = tf.reshape(p_ls, [config.batch_size, config.max_num_digits+1, config.num_glimpses]) # p_ls = (batch_sz, max_digits+1, num_glimpse)

inner_sum = tf.multiply(p_ys, p_ls) # (batch_sz, max_digits+1, num_glimpse)
inner_sum = tf.reduce_sum(inner_sum, axis=2) # (batch_sz, max_digits+1)
log_inner_sum = tf.log(inner_sum) # (batch_sz, max_digits+1)

log_inner_sum_masked = tf.multiply(log_inner_sum, curriculum_ph)

J = tf.reduce_sum(log_inner_sum_masked, axis=1) #(batch_sz,)
J = tf.reduce_mean(J) # MLE objective

# 0/1 rewards.
# Eq(11)
rewards = tf.expand_dims(rewards_ph, axis=-1) # rewards_ph = (batch_sz, max_digits+1), rewards = (batch_sz, max_digits+1, 1)
rewards = tf.tile(rewards, (1, 1, config.num_glimpses))  # (batch_sz, max_digits+1, num_glimpse)
rewards = tf.reshape(rewards, shape=[config.batch_size, timesteps]) # (batch_size, (max_digits+1)*num_glimpse=timestep)


# Eq(12)
baselines = tf.stack(baselines) # (timesteps, batch_size)
baselines = tf.transpose(baselines) # (batch_size, timesteps)

advs = rewards - tf.stop_gradient(baselines)
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std) # (batch_size, timesteps)
logllratio = tf.reduce_mean(logll * advs)
rewards_avg = tf.reduce_mean(tf.reduce_mean(rewards, axis=1), axis=0)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
logging.info(var_list)
logging.info("len = {}".format(len(var_list)))

# hybrid loss
loss = -logllratio -J + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = X_train.shape[0] // config.batch_size
starter_learning_rate = config.lr_start

# decay per training epoch
learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                           global_step=global_step,
                                           decay_steps=training_steps_per_epoch,
                                           decay_rate=0.97,
                                           staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)

opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)

train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)


n_steps = config.step

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_steps):

        images, labels = X_train, Y_train
        # duplicate M times, see Eqn (9)
        images = np.tile(images, [config.M, 1, 1, 1])
        labels = np.tile(labels, [config.M, 1])
        Enet.samping = True

        scores_val = sess.run(
                [scores],
                feed_dict={
                    X_ph: images,
                    Y_ph: labels
                }
        )
        scores_val = np.squeeze(scores_val)
        curriculum_mask_val = build_curriculum(scores_val)
        rewards_mask_val = build_rewards(scores_val)

        Y_pred_val, Y_true_onehot_val, probs_val, p_ys_val, p_ls_val, log_inner_sum_val, log_inner_sum_masked_val, \
        baselines_val, baselines_mse_val, MLE_val, logllratio_val, \
                rewards_val, rewards_avg_val, loss_val, lr_val, _ = sess.run(
                        [Y_pred, Y_true_onehot, probs, p_ys, p_ls, log_inner_sum, log_inner_sum_masked,
                         baselines, baselines_mse, -J, logllratio,
                         rewards, rewards_avg, loss, learning_rate, train_op],
                        feed_dict={
                            X_ph: images,
                            Y_ph: labels,
                            curriculum_ph: curriculum_mask_val,
                            rewards_ph: rewards_mask_val
                        })

        if i and i % 10 == 0:
        # if True:
            logging.debug('set {}: probs = {}, Y_true_onehot = {}'.format(i, probs_val, Y_true_onehot_val))
            logging.debug('set {}: p_ys = {}, p_ls = {}, log_inner_sum = {}, log_inner_sum_masked = {}, curriculum_mask = {}, rewards_mask = {}'.format(i, p_ys_val, p_ls_val, log_inner_sum_val, log_inner_sum_masked_val, curriculum_mask_val, rewards_mask_val))
            logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
            logging.info(
                    'step {}: rewards_avg = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
                            i, rewards_avg_val, loss_val, MLE_val))
            logging.info('llratio = {:3.4f}\tbaselines = {}\trewards = {}\tbaselines_mse = {:3.4f}'.format(
                    logllratio_val, baselines_val, rewards_val, baselines_mse_val))
            # logging.info('log_w = {}'.format(loc_w_val))

        # if i and i % training_steps_per_epoch == 0:
        if True:
            # Evaluation
            y_mask = mask_labels(labels)
            result = np.equal(Y_pred_val * y_mask, labels * y_mask)
            logging.info("step {}: test accuracy = {}".format(i, np.mean(np.min(result, axis=1))))
            logging.debug("step {}: Y = {}, Y_pred = {}".format(i, labels, Y_pred_val))
        #     for dataset in [mnist.validation, mnist.test]:
        #         steps_per_epoch = dataset.num_examples // config.eval_batch_size
        #         correct_cnt = 0
        #         num_samples = steps_per_epoch * config.batch_size
        #         loc_net.sampling = True
        #         for test_step in xrange(steps_per_epoch):
        #             images, labels = dataset.next_batch(config.batch_size)
        #             labels_bak = labels
        #             # Duplicate M times
        #             images = np.tile(images, [config.M, 1])
        #             labels = np.tile(labels, [config.M])
        #             softmax_val = sess.run(softmax,
        #                                      feed_dict={
        #                                              X_ph: images,
        #                                              Y_ph: labels
        #                                      })
        #             softmax_val = np.reshape(softmax_val,
        #                                      [config.M, -1, config.num_classes])
        #             softmax_val = np.mean(softmax_val, 0)
        #             pred_labels_val = np.argmax(softmax_val, 1)
        #             pred_labels_val = pred_labels_val.flatten()
        #             correct_cnt += np.sum(pred_labels_val == labels_bak)
        #         acc = correct_cnt / num_samples
        #         if dataset == mnist.validation:
        #             logging.info('valid accuracy = {}'.format(acc))
        #         else:
        #             logging.info('test accuracy = {}'.format(acc))
