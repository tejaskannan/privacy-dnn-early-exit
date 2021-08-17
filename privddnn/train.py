import os.path
import tensorflow as tf2
from argparse import ArgumentParser
from collections import OrderedDict

from neural_network.base import NeuralNetwork
from neural_network.small_cnn import SmallCNN
from neural_network.large_cnn import LargeCNN

from neural_network.branchynet_cnn import BranchyNetCNN
from neural_network.branchynet_dnn import BranchyNetDNN
from utils.file_utils import read_json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hypers-path', type=str, help='Optional JSON file to override hyperparameters.')
    parser.add_argument('--save-folder', type=str, default='saved_models', help='Folder in which to save the results')
    args = parser.parse_args()

    # Load the data
    (X_train, y_train), _ = tf2.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)

    print('Input Shape: {}, Label Shape: {}'.format(X_train.shape, y_train.shape))

    # Extract hyper-parameters path
    hypers_path: str = args.hypers_path if args.hypers_path is not None else ''
    hypers: OrderedDict = read_json(hypers_path) if len(hypers_path) > 0 and os.path.exists(hypers_path) else OrderedDict()

    dnn = BranchyNetCNN(hypers=hypers)
    dnn.train(inputs=X_train, labels=y_train, save_folder=args.save_folder)
