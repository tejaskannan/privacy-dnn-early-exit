import tensorflow as tf
from tensorflow.keras.metrics import Metric
from typing import Dict

from .base import NeuralNetwork


class EarlyExitNeuralNetwork(NeuralNetwork):

    @property
    def num_outputs(self):
        raise NotImplementedError()

    def make_loss(self) -> Dict[str, str]:
        loss_dict: Dict[str, str] = dict()

        for idx in range(self.num_outputs):
            loss_dict['output{}'.format(idx)] = 'sparse_categorical_crossentropy'

        return loss_dict

    def make_metrics(self) -> Dict[str, Metric]:
        metrics_dict: Dict[str, Metric] = dict()

        for idx in range(self.num_outputs):
            metrics_dict['output{}'.format(idx)] = tf.metrics.SparseCategoricalAccuracy(name='acc')

        return metrics_dict

    def make_loss_weights(self) -> Dict[str, float]:
        loss_weights: Dict[str, float] = dict()

        for idx in range(self.num_outputs):
            loss_weights['output{}'.format(idx)] = (idx + 1) / self.num_outputs

        return loss_weights
