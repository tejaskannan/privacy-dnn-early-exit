import tensorflow as tf2
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.FINE_TUNE)

    # Get the dataset
    (train_inputs, train_labels), _ = tf2.keras.datasets.cifar10.load_data()
    train_labels = train_labels.reshape(-1)  # [B]

    # Use larger batches to have better rate averaging
    model.hypers['batch_size'] = 100
    model.hypers['num_epochs'] = 10

    # Get the predictions from the models
    model.train(train_inputs, train_labels, model_mode=ModelMode.FINE_TUNE, save_folder='tuned_models')
