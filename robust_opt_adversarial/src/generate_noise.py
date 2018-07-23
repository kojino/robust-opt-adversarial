import argparse
import logging
import os

import numpy as np

from noise import Noise
from utils import load_data

logger = logging.getLogger("generate_noise")
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    level='INFO',
    datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--test_eval', type=bool, default=False)
args = parser.parse_args()

noises = [
    'vert_shrink25', 'horiz_shrink25', 'both_shrink25', 'light_tint',
    'gradient', 'checkerboard', 'pos_noise', 'mid_noise', 'neg_noise'
]

# Load data
X_train, Y_train, X_test, Y_test = load_data(args.data)

# Flatten image matrices
num_train, num_rows, num_cols, num_channels = X_train.shape
num_test, _, _, _ = X_test.shape
_, num_classes = Y_train.shape

direct_noise = Noise()
path = f'../data/{args.data}/noise'
os.makedirs(path, exist_ok=True)
for noise in noises:
    X_train_noise = direct_noise.apply_noise(X_train, noise)
    np.save(f'{path}/{noise}.npy', X_train_noise)
    logger.info(f"Saved noise {noise}")
