import os.path
import tensorflow as tf2
from argparse import ArgumentParser
from collections import OrderedDict

from neural_network import make_model, NeuralNetwork, ModelMode
from utils.file_utils import read_json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-name', type=str, help='Name of the neural network model to train.')
    parser.add_argument('--dataset-name', type=str, help='Name of the dataset.', choices=['mnist', 'fashion_mnist', 'cifar_10'])
    parser.add_argument('--hypers-path', type=str, help='Optional JSON file to override hyperparameters.')
    parser.add_argument('--save-folder', type=str, default='saved_models', help='Folder in which to save the results')
    args = parser.parse_args()

    # Extract hyper-parameters path
    hypers_path: str = args.hypers_path if args.hypers_path is not None else ''
    hypers: OrderedDict = read_json(hypers_path) if len(hypers_path) > 0 and os.path.exists(hypers_path) else OrderedDict()

    dnn: NeuralNetwork = make_model(name=args.model_name, dataset_name=args.dataset_name, hypers=hypers)
    dnn.train(save_folder=args.save_folder, model_mode=ModelMode.TRAIN)
