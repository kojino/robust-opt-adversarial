import math

import numpy as np

import keras
import tensorflow as tf
from cleverhans.attacks import DeepFool
from cleverhans.utils import batch_indices
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_mnist import data_mnist
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

lr = 0.001
epochs = 10
batch_size = 128
drop_ratio = 0.10
num_inputs = int(28 * 28 * (1 - drop_ratio))
noise_eps = 0.1


def logistic_regression_model(input_ph, num_inputs, nb_classes=10):
    model = Sequential()
    model.add(Dense(nb_classes, input_shape=(num_inputs, )))
    model.add(Activation('softmax'))
    return model


# Create TF session and set as Keras backend session
sess = tf.Session()
# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist()
# Flatten image matrices
num_train, num_rows, num_cols, num_channels = X_train.shape
num_test, _, _, _ = X_test.shape
X_train = X_train.reshape((num_train, num_rows * num_cols * num_channels))
X_test = X_test.reshape((num_test, num_rows * num_cols * num_channels))

# Randomly drop some ratio of pixels
input_indices = np.random.choice(
    num_rows * num_cols, num_inputs, replace=False)
X_train = X_train[:, input_indices]
X_test = X_test[:, input_indices]

# Define placeholders
x = tf.placeholder(tf.float32, shape=(None, num_inputs))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Wrap log reg model for applying adversarial examples
model = logistic_regression_model(x, num_inputs)
wrap = KerasModelWrapper(model, 10)

# Define the objective
logits = wrap.get_logits(x)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)

# Define the update func
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss)
acc, acc_op = tf.metrics.accuracy(
    labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

# Define adv attack
deepfool = DeepFool(wrap, sess=sess)
deepfool_params = {'eps': noise_eps, 'clip_min': 0., 'clip_max': 1.}

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

for epoch in range(epochs):
    # Compute number of batches
    num_batches = int(math.ceil(float(len(X_train)) / batch_size))
    assert num_batches * batch_size >= len(X_train)

    # Indices to shuffle training set
    index_shuf = list(range(len(X_train)))
    rng.shuffle(index_shuf)
    for batch in range(num_batches):

        # Compute batch start and end indices
        start, end = batch_indices(batch, len(X_train), batch_size)

        # Perform one training step
        feed_dict = {
            x: X_train[index_shuf[start:end]],
            y: Y_train[index_shuf[start:end]]
        }
        sess.run(train_step, feed_dict=feed_dict)

    feed_dict = {x: X_train, y: Y_train}
    acc_val = sess.run(acc_op, feed_dict=feed_dict)
    print(f"Epoch: {epoch}, Train Acc: {acc_val:.5f}")

# Evaluate the model on the test set
feed_dict = {x: X_test, y: Y_test}
test_acc = sess.run(acc_op, feed_dict=feed_dict)
print(f"Test Acc: {test_acc}")

# Evaluate the model on the adversarial test set
test_acc_adv = sess.run(acc_op_adv, feed_dict=feed_dict)
print(f"Test Acc Adv: {test_acc_adv}")
