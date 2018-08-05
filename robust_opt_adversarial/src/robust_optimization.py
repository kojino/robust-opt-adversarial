import argparse
import copy
import logging
import math
import os
import datetime
import resource

import keras
import numpy as np
import tensorflow as tf

from cleverhans.attacks import DeepFool, MomentumIterativeMethod
from cleverhans.utils import batch_indices
from cleverhans.utils_keras import KerasModelWrapper
from utils import (fully_connected_nn_model, load_data, to_onehot)

logger = logging.getLogger("robust_optimization")
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    level='INFO',
    datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser()
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--noise_eps', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_noises', type=int, default=3)
parser.add_argument('--num_oracle_iter', type=int, default=2)
parser.add_argument(
    '--data', type=str, choices=['mnist', 'cifar10'], default='mnist')
args = parser.parse_args()
args.num_noises += 1  # for the true data

start_time = datetime.datetime.now()
logger.info(f"START TIME {start_time}")

# Load data
X_train, Y_train, X_test, Y_test = load_data(args.data)
num_train, num_rows, num_cols, num_channels = X_train.shape
_, num_classes = Y_train.shape

# X_test_padded will be fed into the graph under x
# so it needs to have the same shape including the first dimensionsion args.num_noises
# even though we only need x[0] to compute accuracy
# this is a waste of memory but is needed to make it work if we write the computational graph this way

X_test_padded = np.array([X_test for _ in range(args.num_noises)]).reshape(args.num_noises, 10000, 28, 28, 1)


logger.info(f"num_train {num_train}, num_rows {num_rows}, num_cols {num_cols}, num_channels {num_channels}")
logger.info(f"num_classes {num_classes}")

# Load adversarial examples generated from logistic regressions
X_train_adv = np.zeros((args.num_noises, num_train, num_rows, num_cols,
                        num_channels))
X_train_adv[0] = X_train
for i in range(1, args.num_noises):
    X_train_adv[i] = np.load(f"../data/{args.data}/logreg_adv/adv_{i-1}.npy")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = tf.placeholder(
    tf.float32,
    shape=(args.num_noises, None, num_rows, num_cols, num_channels))

y = tf.placeholder(tf.float32, shape=(None, num_classes))
w = tf.placeholder(tf.float32, shape=(args.num_noises, ))

model = fully_connected_nn_model(
    (num_rows, num_cols, num_channels),
    num_classes=num_classes,
    num_hidden_layers=args.num_hidden_layers)
wrap = KerasModelWrapper(model)

losses = tf.zeros([0], tf.float32)


def body(i, losses):
    logits = wrap.get_logits(x[i])
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    losses = tf.concat([losses, [loss]], 0)
    return i + 1, losses


def condition(i, losses):
    return i < args.num_noises


_, losses = tf.while_loop(
    condition,
    body, [0, losses],
    shape_invariants=[tf.TensorShape(None),
                      tf.TensorShape([None])])

# Define the update func
loss = w * losses
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
train_step = optimizer.minimize(loss)

# Test acc on legit data
logits = wrap.get_logits(x[0])
acc, acc_op = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

# Define adv attack
deepfool = DeepFool(wrap, sess=sess)
deepfool_params = {'eps': args.noise_eps, 'clip_min': 0., 'clip_max': 1.}

# Attack images
x_deepfool = deepfool.generate(x[0], **deepfool_params)
# Consider the attack to be constant
x_deepfool = tf.stop_gradient(x_deepfool)

# Evaluate predictions on adv attacks
preds_deepfool = model(x_deepfool)
acc_deepfool, acc_op_deepfool = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(preds_deepfool, 1))

# Define adv attack
momentum_iterative = MomentumIterativeMethod(wrap, sess=sess)
momentum_iterative_params = {
    'eps': args.noise_eps,
    'clip_min': 0.,
    'clip_max': 1.
}

# Attack images
x_momentum_iterative = momentum_iterative.generate(x[0], **deepfool_params)
# Consider the attack to be constant
x_momentum_iterative = tf.stop_gradient(x_momentum_iterative)

# Evaluate predictions on adv attacks
preds_momentum_iterative = model(x_momentum_iterative)
acc_momentum_iterative, acc_op_momentum_iterative = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(preds_momentum_iterative, 1))

saver = tf.train.Saver()
rng = np.random.RandomState()  # for batch sampling

# Each row is w_t, simplex vector over noises
weights_distribution = np.full((args.num_oracle_iter, args.num_noises),
                               1. / args.num_noises)
# Has value L_i(x_t) for each i in noises and t in oracle_iter
losses_np = np.zeros((args.num_oracle_iter, args.num_noises))

# Number of times the Bayesian oracle is invoked
for oracle_iter in range(args.num_oracle_iter):
    logging.info(f'oracle_iter: {oracle_iter}')
    # initialize variables every iteration
    sess.run(tf.initialize_all_variables())

    ### Compute the weights for the distributional oracle for this iteration ###
    # eta is the time dependent parameter controlling the weight distribution
    eta = np.sqrt(np.log(args.num_noises) / (2 * args.num_oracle_iter))
    # See Algorithm 1 (3) for this udpate
    unnormalized_current_weights = np.exp(
        eta * losses_np[0:oracle_iter, :].sum(axis=0))
    normalized_current_weights = unnormalized_current_weights / np.sum(
        unnormalized_current_weights)
    # Log current weights for later iterations
    weights_distribution[oracle_iter, :] = normalized_current_weights
    logging.info(f"Noise ratio: {normalized_current_weights}")

    ### Train model with weighted loss for each noise ###
    loss_total_list = np.zeros((args.num_noises, ))
    for epoch in range(args.epochs):
        logging.info(f'epoch: {epoch}')
        # Compute number of batches
        num_batches = int(math.ceil(num_train / args.batch_size))
        assert num_batches * args.batch_size >= num_train

        # Indices to shuffle training set
        index_shuf = list(range(num_train))
        rng.shuffle(index_shuf)
        for batch in range(num_batches):
            # Compute batch start and end indices
            start, end = batch_indices(batch, num_train, args.batch_size)
            feed_dict = {
                x: X_train_adv[:, index_shuf[start:end]],
                y: Y_train[index_shuf[start:end]],
                w: weights_distribution[oracle_iter]
            }
            loss_vals, _ = sess.run([losses, train_step], feed_dict=feed_dict)
            # Normalize and log loss
            loss_total_list += loss_vals / (args.epochs * num_batches)
    # Save average loss for the next iteration
    losses_np[oracle_iter, :] = loss_total_list
    # Save the lastest trained model
    os.makedirs("../model/robust_optimization/", exist_ok=True)
    saver.save(sess, "../model/robust_optimization/model.ckpt")

# solve https://stackoverflow.com/questions/46079644/tensorflow-attempting-to-use-uninitialized-value-error-when-restoring
# "The accuracy operation contains some local variable which is not part of the graph, 
# so it should be initialized manually. adding sess.run(tf.local_variables_initializer()) 
# after restore will initialize the local variables."
sess.run(tf.local_variables_initializer())

# 1. Accuracy on uncorrupted test set
feed_dict = {x: X_test_padded, 
             y: Y_test}

test_acc = sess.run(acc_op, feed_dict=feed_dict)

logger.info(f"Test Acc: {test_acc}")

# 2. Accuracy on test set corrupted by known noises that are used during training
test_acc_deepfool = sess.run(acc_op_deepfool, feed_dict=feed_dict)
logger.info(f"Test Acc Deepfool: {test_acc_deepfool}")

# 3. Accuracy on test set corrupted by adversarial noises not used during training
test_acc_momentum_iterative = sess.run(
    acc_op_momentum_iterative, feed_dict=feed_dict)
logger.info(f"Test Acc Momentum Iterative: {test_acc_momentum_iterative}")

# Log experiment result
path = f'../result/robust_optimization'
os.makedirs(path, exist_ok=True)
losses_np.dump(f'{path}/losses.npy')
weights_distribution.dump(f'{path}/weights_distribution.npy')
accs = np.array([test_acc, test_acc_deepfool, test_acc_momentum_iterative])
accs.dump(f'{path}/accs.npy')

finish_time = datetime.datetime.now()
logger.info(f"START   TIME {start_time}")
logger.info(f"FINISH  TIME {finish_time}")
logger.info(f"ELAPSED TIME {finish_time-start_time}")

logger.info(f"PEAK MEMORY USAGE {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KILOBYTES")

logger.info(f"PARAMETER SUMMARIES")
for k,v in sorted(vars(args).items()):
    logger.info("{0}: {1}".format(k,v))

logger.info("TEST SUMMARIES")
logger.info(f"Test Acc: {test_acc}")
logger.info(f"Test Acc Deepfool: {test_acc_deepfool}")
logger.info(f"Test Acc Momentum Iterative: {test_acc_momentum_iterative}")

