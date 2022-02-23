import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import os.path
import h5py
from collections import OrderedDict

from privddnn.classifier import ModelMode, OpName
from .base import NeuralNetwork
from .layers import conv1d, dense


class BranchyNet1dCNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-1dcnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        # Create the convolutions
        input_shape = inputs.get_shape()

        # Make the 1D CNN blocks
        conv0 = conv1d(inputs=inputs,
                       filter_size=32,
                       num_filters=32,
                       stride=2,
                       activation='relu',
                       trainable=True,
                       name='conv0')

        conv1 = conv1d(inputs=conv0,
                       filter_size=32,
                       num_filters=64,
                       stride=2,
                       activation='relu',
                       trainable=True,
                       name='conv1')

        conv2 = conv1d(inputs=conv1,
                       filter_size=16,
                       num_filters=128,
                       stride=1,
                       activation='relu',
                       trainable=True,
                       name='conv2')

        conv3 = conv1d(inputs=conv2,
                       filter_size=8,
                       num_filters=128,
                       stride=1,
                       activation='relu',
                       trainable=True,
                       name='conv3')

        conv4 = conv1d(inputs=conv3,
                       filter_size=8,
                       num_filters=128,
                       stride=1,
                       activation='relu',
                       trainable=True,
                       name='conv4')

        # Make the output layers
        conv0_shape = conv0.get_shape()
        flattened0 = tf2.reshape(conv0, (-1, np.prod(conv0_shape[1:])))

        output0 = dense(inputs=flattened0,
                        output_units=num_labels,
                        use_dropout=False,
                        dropout_keep_rate=1.0,
                        activation='linear',
                        trainable=True,
                        name='output0')  # [B, K]

        conv4_shape = conv4.get_shape()
        flattened4 = tf2.reshape(conv4, (-1, np.prod(conv4_shape[1:])))

        output1_hidden0 = dense(inputs=flattened4,
                                output_units=1024,
                                use_dropout=True,
                                dropout_keep_rate=dropout_keep_rate,
                                activation='relu',
                                trainable=True,
                                name='output1-hidden0')  # [B, D]

        output1_hidden1 = dense(inputs=output1_hidden0,
                                output_units=1024,
                                use_dropout=True,
                                dropout_keep_rate=dropout_keep_rate,
                                activation='relu',
                                trainable=True,
                                name='output1-hidden1')  # [B, D]

        output1 = dense(inputs=output1_hidden1,
                        output_units=num_labels,
                        use_dropout=False,
                        dropout_keep_rate=1.0,
                        activation='linear',
                        trainable=True,
                        name='output1')  # [B, D]

        logits = tf2.concat([tf2.expand_dims(output0, axis=1), tf2.expand_dims(output1, axis=1)], axis=1)  # [B, 2, K]
        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        # Tile the labels along an expanded axis
        labels = tf2.expand_dims(labels, axis=1)
        labels = tf2.tile(labels, multiples=(1, self.num_outputs))  # [B, 2]

        # Compute the per-output loss and aggregate for each sample
        sample_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, 2]
        aggregate_sample_loss = tf2.reduce_sum(sample_loss * tf2.constant([0.0, 1.0], dtype=sample_loss.dtype), axis=-1)  # [B]

        return tf2.reduce_mean(aggregate_sample_loss)
