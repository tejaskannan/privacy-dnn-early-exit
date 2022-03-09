import os.path
from collections import OrderedDict
from typing import Type

from privddnn.classifier import OpName, ModelMode
from .base import NeuralNetwork
from .vgg import VGG
from .branchynet_cnn import BranchyNetCNN, BranchyNetCNN3, BranchyNetCNN4
from .branchynet_dnn import BranchyNetDNN, BranchyNetDNN3, BranchyNetDNN4
from .branchynet_rnn import BranchyNetRNN
from .speech_cnn import SpeechCNN, SpeechCNN3, SpeechCNN4


def get_model_class(name: str) -> Type[NeuralNetwork]:
    name = name.lower()

    if name == 'branchynet-cnn':
        return BranchyNetCNN
    elif name == 'branchynet-cnn-3':
        return BranchyNetCNN3
    elif name == 'branchynet-cnn-4':
        return BranchyNetCNN4
    elif name == 'branchynet-dnn':
        return BranchyNetDNN
    elif name == 'branchynet-dnn-3':
        return BranchyNetDNN3
    elif name == 'branchynet-dnn-4':
        return BranchyNetDNN4
    elif name == 'vgg':
        return VGG
    elif name == 'speech-cnn':
        return SpeechCNN
    elif name == 'speech-cnn-3':
        return SpeechCNN3
    elif name == 'speech-cnn-4':
        return SpeechCNN4
    else:
        raise ValueError('Unknown neural network with name: {}'.format(name))


def make_model(name: str, dataset_name: str, hypers: OrderedDict) -> NeuralNetwork:
    model_cls = get_model_class(name=name)
    return model_cls(hypers=hypers, dataset_name=dataset_name)


def restore_model(path: str, model_mode: ModelMode) -> NeuralNetwork:
    # Extract the model name
    file_name = os.path.split(path)[-1]
    components = file_name.split('_')
    model_name = components[0].lower()

    # Restore the model
    model_cls = get_model_class(name=model_name)
    return model_cls.restore(path, model_mode=model_mode)
