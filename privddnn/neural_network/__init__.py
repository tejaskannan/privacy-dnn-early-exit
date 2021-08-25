import os.path
from collections import OrderedDict

from .base import NeuralNetwork
from .branchynet_cnn import BranchyNetCNN
from .constants import OpName, ModelMode


def make_model(name: str, dataset_name: str, hypers: OrderedDict) -> NeuralNetwork:
    name = name.lower()

    if name == 'branchynet-cnn':
        return BranchyNetCNN(hypers=hypers, dataset_name=dataset_name)
    else:
        raise ValueError('Unknown neural network with name: {}'.format(name))


def restore_model(path: str, model_mode: ModelMode) -> NeuralNetwork:
    # Extract the model name
    file_name = os.path.split(path)[-1]
    components = file_name.split('_')
    model_name = components[0].lower()

    # Restore the model
    if model_name == 'branchynet-cnn':
        model_cls = BranchyNetCNN
    else:
        raise ValueError('Unknown neural network with name: {}'.format(model_name))

    return model_cls.restore(path, model_mode=model_mode)
