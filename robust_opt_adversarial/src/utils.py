import numpy as np
from keras.datasets import cifar10
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

from cleverhans.utils_mnist import data_mnist


def logistic_regression_model(num_inputs, num_classes=10):
    model = Sequential()
    model.add(Dense(num_classes, input_shape=(num_inputs, )))
    model.add(Activation('softmax'))
    return model


def fully_connected_nn_model(input_shape,
                             num_classes=10,
                             num_hidden_size=1024,
                             num_hidden_layers=1):
    if num_hidden_layers == 0:
        raise ValueError("Use logistic_regression_model instead.")
    model = Sequential()
    for i in range(num_hidden_layers):
        if i == 0:
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(num_hidden_size))
        else:
            model.add(Dense(num_hidden_size))
    model.add(Dense(num_classes))
    return model


def to_onehot(vec):
    vec = vec.flatten()
    vec_new = np.zeros((len(vec), np.max(vec) + 1))
    vec_new[np.arange(len(vec)), vec] = 1
    return vec_new


def load_data(data='mnist'):
    if data == 'mnist':
        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()
    elif data == 'cifar10':
        # Get Cifar10 data
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        Y_train = to_onehot(Y_train.flatten())
        Y_test = to_onehot(Y_test.flatten())
    else:
        raise ValueError("Dataset not compatible.")
    return X_train, Y_train, X_test, Y_test
