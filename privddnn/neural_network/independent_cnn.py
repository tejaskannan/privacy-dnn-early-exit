import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import os.path
import h5py
from collections import OrderedDict

from privddnn.classifier import ModelMode, OpName
from .base import NeuralNetwork
from .layers import conv2d, dense, dropout, fitnet_block


class IndependentCNN(NeuralNetwork):

    def __init__(self, dataset_name: str, hypers: OrderedDict):
        super().__init__(dataset_name=dataset_name, hypers=hypers)
        assert dataset_name == 'cifar_10', 'Independent CNNs only support Cifar-10'

        # Get the paths to the (precomputed) larger model predictions on the validation and testing sets
        base = os.path.dirname(os.path.abspath(__file__))
        self._val_path = os.path.join(base, 'pretrained', 'vggcifar10_val.h5')
        self._test_path = os.path.join(base, 'pretrained', 'vggcifar10_test.h5')

        self._val_probs = np.empty(shape=(1, ))
        self._val_preds = np.empty(shape=(1, ))
        self._test_probs = np.empty(shape=(1, ))
        self._test_preds = np.empty(shape=(1, ))

        self._is_val_loaded = False
        self._is_test_loaded = False

    @property
    def name(self) -> str:
        return 'independent-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        # Create the convolutions
        conv1_block, _, _ = fitnet_block(inputs=inputs, num_filters=16, pool_size=4, pool_stride=2, name='block1', trainable=True)
        conv2_block, _, _ = fitnet_block(inputs=conv1_block, num_filters=16, pool_size=4, pool_stride=2, name='block2', trainable=True)
        conv3_block, _, _ = fitnet_block(inputs=conv2_block, num_filters=12, pool_size=2, pool_stride=1, name='block3', trainable=True)

        # Flatten the output
        conv3_block_shape = conv3_block.get_shape()
        flattened_two = tf2.reshape(conv3_block, (-1, np.prod(conv3_block_shape[1:])))

        logits = dense(inputs=flattened_two,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=True,
                           name='output')  # [B, K]

        return logits

    def _load_test_preds(self):
        if self._is_test_loaded:
            return

        with h5py.File(self._test_path, 'r') as fin:
            self._test_preds = fin['preds'][:]
            self._test_probs = fin['probs'][:]

        self._is_test_loaded = True

    def _load_val_preds(self):
        if self._is_val_loaded:
            return

        with h5py.File(self._val_path, 'r') as fin:
            self._val_preds = fin['preds'][:]
            self._val_probs = fin['probs'][:]

        self._is_val_loaded = True

    def test(self, op: OpName) -> np.ndarray:
        assert op in (OpName.PROBS, OpName.PREDICTIONS), 'Op must be Probs or Predictions'

        first_preds = super().test(op)  # [B, K] or [B]
        first_preds = np.expand_dims(first_preds, axis=1)  # [B, 1, K] or [B, 1]

        self._load_test_preds()

        if op == OpName.PREDICTIONS:
            larger_preds = np.expand_dims(self._test_preds, axis=1)  # [B, 1]
        else:
            larger_preds = np.expand_dims(self._test_probs, axis=1)  # [B, 1, K]

        return np.concatenate([first_preds, larger_preds], axis=1)  # [B, 2, K] or [B, 2]

    def validate(self, op: OpName) -> np.ndarray:
        assert op in (OpName.PROBS, OpName.PREDICTIONS), 'Op must be Probs or Predictions'

        first_preds = super().validate(op)  # [B, K] or [B]
        first_preds = np.expand_dims(first_preds, axis=1)  # [B, 1, K] or [B, 1]

        self._load_val_preds()

        if op == OpName.PREDICTIONS:
            larger_preds = np.expand_dims(self._val_preds, axis=1)  # [B, 1]
        else:
            larger_preds = np.expand_dims(self._val_probs, axis=1)  # [B, 1, K]

        return np.concatenate([first_preds, larger_preds], axis=1)  # [B, 2, K] or [B, 2]

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        sample_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf2.reduce_mean(sample_loss)
