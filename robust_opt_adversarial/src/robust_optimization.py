import copy
import math
import os
import argparse
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

from cleverhans.attacks import DeepFool
from cleverhans.utils import batch_indices
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_mnist import data_mnist
from utils import load_data, logistic_regression_model, to_onehot, fully_connected_nn_model

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_noises', type=int, default=3)
parser.add_argument('--num_oracle_iter', type=int, default=2)
parser.add_argument(
    '--data', type=str, choices=['mnist', 'cifar10'], default='mnist')
args = parser.parse_args()

X_train, Y_train, X_test, Y_test = load_data('mnist')
num_train, num_rows, num_cols, num_channels = X_train.shape
_, num_classes = Y_train.shape

X_train_adv = np.zeros((args.num_noises, num_train, num_rows, num_cols,
                        num_channels))
for i in range(args.num_noises):
    X_train_adv[i] = np.load(f"../data/mnist/adv_{i}.npy")

# each row is w_t, simplex vector over noises
weights_distribution = np.full((args.num_oracle_iter, args.num_noises),
                               1. / args.num_noises)
# has value L_i(x_t) for each i in noises and t in oracle_iter
losses = np.zeros((args.num_oracle_iter, args.num_noises))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
w = tf.placeholder(tf.float32)

model = fully_connected_nn_model((num_rows, num_cols, num_channels))
logits = model(x)
loss = w * tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

# Define the update func
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
epochs = 2
batch_size = 128
rng = np.random.RandomState()  # for batch sampling
# number of times the Bayesian oracle is invoked
for oracle_iter in range(args.num_oracle_iter):
    print(f'oracle_iter: {oracle_iter}')
    ### Compute the weights for the distributional oracle for this iteration ###
    # eta is the time dependent parameter controlling the weight distribution
    eta = np.sqrt(np.log(args.num_noises) / (2 * args.num_oracle_iter))
    # See Algorithm 1 (3) for this udpate
    unnormalized_current_weights = np.exp(
        eta * losses[0:oracle_iter, :].sum(axis=0))
    print(unnormalized_current_weights)
    normalized_current_weights = unnormalized_current_weights / np.sum(
        unnormalized_current_weights)
    # Log current weights for later iterations
    weights_distribution[oracle_iter, :] = normalized_current_weights

    loss_total_list = np.zeros((args.num_noises, ))
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        # Compute number of batches
        num_batches = int(math.ceil(num_train / batch_size))
        assert num_batches * batch_size >= num_train

        # Indices to shuffle training set
        index_shuf = list(range(num_train))
        rng.shuffle(index_shuf)
        for batch in range(num_batches):
            # Compute batch start and end indices
            start, end = batch_indices(batch, num_train, batch_size)

            for noise_type in range(args.num_noises):
                # Perform one training step
                feed_dict = {
                    x: X_train_adv[noise_type][index_shuf[start:end]],
                    y: Y_train[index_shuf[start:end]],
                    w: weights_distribution[oracle_iter, noise_type]
                }
                loss_val, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                loss_total_list[
                    noise_type] += loss_val / (epochs * num_batches)
    print(loss_total_list)
    losses[oracle_iter, :] = loss_total_list
