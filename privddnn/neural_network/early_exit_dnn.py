import tensorflow as tf
from tensorflow.keras.metrics import Metric
from typing import Dict

from .base import NeuralNetwork


class EarlyExitNeuralNetwork(NeuralNetwork):

    @property
    def num_outputs(self):
        raise NotImplementedError()

    def make_loss(self) -> Dict[str, str]:
        return {
            'output0': 'sparse_categorical_crossentropy',
            'output1': 'sparse_categorical_crossentropy'
        }

    def make_metrics(self) -> Dict[str, Metric]:
        return {
            'output0': tf.metrics.SparseCategoricalAccuracy(name='acc'),
            'output1': tf.metrics.SparseCategoricalAccuracy(name='acc')
        }

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.25,
            'output1': 0.75
        }
