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
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.FINE_TUNE)

    # Use larger batches to have better rate averaging
    model.hypers['batch_size'] = 100
    model.hypers['num_epochs'] = 10

    # Get the predictions from the models
    model.train(model_mode=ModelMode.FINE_TUNE, save_folder='tuned_models')
