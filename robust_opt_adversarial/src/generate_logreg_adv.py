import argparse
import copy
import logging
import math
import os
import sys
import datetime

import keras
import numpy as np
import tensorflow as tf

# tf.logging.set_verbosity('tf.logging.WARN')

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import batch_indices
from cleverhans.utils_keras import KerasModelWrapper
from utils import load_data, logistic_regression_model

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


logger = logging.getLogger("noise_against_log_reg")

logging.basicConfig(
   format='%(asctime)s: %(message)s',
   level='INFO',
   stream=sys.stderr,
   datefmt='%m/%d/%Y %I:%M:%S %p')


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
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

# fh = logging.FileHandler(f'noise_against_log_reg_{args.seed}.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

start_time = datetime.datetime.now()
logger.info(f"START TIME {start_time}")


logger.info(f"Generating adversarial examples for {args.data}, seed {args.seed}")

# Create TF session and set as Keras backend session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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
wrap = KerasModelWrapper(model)

# Define the objective
logits = wrap.get_logits(x)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)

# Define the update func
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
train_step = optimizer.minimize(loss)

if args.verbose:
    print('verbose: train_step done')

acc, acc_op = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

if args.verbose:
    print('verbose: acc={}, acc_op={}'.format(acc, acc_op))

# Define adv attack
deepfool = FastGradientMethod(wrap, sess=sess)
deepfool_params = {'eps': args.noise_eps, 'clip_min': 0., 'clip_max': 1.}

if args.verbose:
    print('verbose: about to generate deepfool noise')

# Attack images
adv_x = deepfool.generate(x, **deepfool_params)

if args.verbose:
    print('verbose: adv_x generate done')

# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)

if args.verbose:
    print('verbose: tf.stop_gradient done')

# Evaluate predictions on adv attacks
preds = model(adv_x)
acc_adv, acc_op_adv = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(preds, 1))

if args.verbose:
    print('verbose: acc_adv={}, acc_op_adv={}'.format(acc_adv, acc_op_adv))

# Initialize variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
rng = np.random.RandomState()  # for batch sampling

if args.verbose:
    print("verbose: about to enter epoch for loop")

for epoch in range(args.epochs):
    # Compute number of batches
    num_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
    assert num_batches * args.batch_size >= len(X_train)

    # Indices to shuffle training set
    index_shuf = list(range(len(X_train)))
    rng.shuffle(index_shuf)

    if args.verbose:
        print("verbose: in epoch for loop, about to do batch loop")

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
    logger.info(f"Epoch: {epoch}, Train Acc: {acc_val:.5f}")

if args.verbose:
    print("verbose: epoch loop done")

if args.test_eval:
    # Evaluate the model on the test set
    if args.verbose:
        print("verbose: evaluate model on test set")

    feed_dict = {x: X_test, y: Y_test}
    test_acc = sess.run(acc_op, feed_dict=feed_dict)
    logger.info(f"Test Acc: {test_acc}")

    if args.verbose:
        print("verbose: test_acc={}".format(test_acc))

    # Evaluate the model on the adversarial test set
    test_acc_adv = sess.run(acc_op_adv, feed_dict=feed_dict)
    logger.info(f"Test Acc Adv: {test_acc_adv}")

    if args.verbose:
        print("verbose: test_acc_adv={}".format(test_acc_adv))

if args.verbose:
    print("verbose: sess.run adv_x_np finished")

logger.info("Generating noise...")
adv_x_np = sess.run(adv_x, feed_dict=feed_dict)
logger.info("Generating noise done!")
logger.info("Saving noise...")

advf_x_np = copy.deepcopy(X_train_org)
advf_x_np[:, input_indices] = adv_x_np
advf_x_np = advf_x_np.reshape(num_train, num_rows, num_cols, num_channels)

if args.verbose:
    print("verbose: advf_x_np created")

path = f'../data/{args.data}/logreg_adv'
os.makedirs(path, exist_ok=True)
np.save(f'{path}/adv_{args.seed}.npy', advf_x_np)
np.save(f'{path}/ind_{args.seed}.npy', input_indices)
logger.info(f"Saved adversarial examples for {args.data}, seed {args.seed}")

finish_time = datetime.datetime.now()
logger.info(f"START   TIME {start_time}")
logger.info(f"FINISH  TIME {finish_time}")
logger.info(f"ELAPSED TIME {finish_time-start_time}")
