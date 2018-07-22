import argparse
import copy
import logging
import math
import os

import keras
import numpy as np
import tensorflow as tf

from cleverhans.attacks import DeepFool
from cleverhans.utils import batch_indices
from cleverhans.utils_keras import KerasModelWrapper
from utils import load_data, logistic_regression_model, to_onehot

logger = logging.getLogger("noise_against_log_reg")
logging.basicConfig(
    format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--drop_ratio', type=float, default=0.5)
parser.add_argument('--noise_eps', type=float, default=0.1)
parser.add_argument(
    '--data', type=str, choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--test_eval', type=bool, default=False)
args = parser.parse_args()

logger.INFO(
    f"Generating adversarial examples for {args.data}, seed {args.seed}")

# Create TF session and set as Keras backend session
sess = tf.Session()

# Load data
X_train_org, Y_train, X_test_org, Y_test = load_data(args.data)

# Flatten image matrices
num_train, num_rows, num_cols, num_channels = X_train_org.shape
num_test, _, _, _ = X_test_org.shape
_, num_classes = Y_train.shape
X_train_org = X_train_org.reshape((num_train,
                                   num_rows * num_cols * num_channels))
X_test_org = X_test_org.reshape((num_test, num_rows * num_cols * num_channels))
num_elements = num_rows * num_cols * num_channels
num_inputs = int(num_elements * (1 - args.drop_ratio))

# Randomly drop some ratio of pixels
np.random.seed(args.seed)
input_indices = np.random.choice(num_elements, num_inputs, replace=False)
X_train = X_train_org[:, input_indices]
X_test = X_test_org[:, input_indices]

# Define placeholders
x = tf.placeholder(tf.float32, shape=(None, num_inputs))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

# Wrap log reg model for applying adversarial examples
model = logistic_regression_model(num_inputs)
wrap = KerasModelWrapper(model, num_classes)

# Define the objective
logits = wrap.get_logits(x)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)

# Define the update func
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
train_step = optimizer.minimize(loss)
acc, acc_op = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

# Define adv attack
deepfool = DeepFool(wrap, sess=sess)
deepfool_params = {'eps': args.noise_eps, 'clip_min': 0., 'clip_max': 1.}

# Attack images
adv_x = deepfool.generate(x, **deepfool_params)
# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)

# Evaluate predictions on adv attacks
preds = model(adv_x)
acc_adv, acc_op_adv = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(preds, 1))

# Initialize variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
rng = np.random.RandomState()  # for batch sampling

for epoch in range(args.epochs):
    # Compute number of batches
    num_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
    assert num_batches * args.batch_size >= len(X_train)

    # Indices to shuffle training set
    index_shuf = list(range(len(X_train)))
    rng.shuffle(index_shuf)
    for batch in range(num_batches):

        # Compute batch start and end indices
        start, end = batch_indices(batch, len(X_train), args.batch_size)

        # Perform one training step
        feed_dict = {
            x: X_train[index_shuf[start:end]],
            y: Y_train[index_shuf[start:end]]
        }
        sess.run(train_step, feed_dict=feed_dict)

    feed_dict = {x: X_train, y: Y_train}
    acc_val = sess.run(acc_op, feed_dict=feed_dict)
    logger.INFO(f"Epoch: {epoch}, Train Acc: {acc_val:.5f}")

if args.test_eval:
    # Evaluate the model on the test set
    feed_dict = {x: X_test, y: Y_test}
    test_acc = sess.run(acc_op, feed_dict=feed_dict)
    logger.INFO(f"Test Acc: {test_acc}")

    # Evaluate the model on the adversarial test set
    test_acc_adv = sess.run(acc_op_adv, feed_dict=feed_dict)
    logger.INFO(f"Test Acc Adv: {test_acc_adv}")

adv_x_np = sess.run(adv_x, feed_dict=feed_dict)
advf_x_np = copy.deepcopy(X_train_org)
advf_x_np[:, input_indices] = adv_x_np
advf_x_np = advf_x_np.reshape(num_train, num_rows, num_cols, num_channels)

path = f'../data/{args.data}'
os.makedirs(path, exist_ok=True)
np.save(f'../data/{args.data}/adv_{args.seed}.npy', advf_x_np)
np.save(f'../data/{args.data}/ind_{args.seed}.npy', input_indices)
logger.INFO(f"Saved adversarial examples for {args.data}, seed {args.seed}")
