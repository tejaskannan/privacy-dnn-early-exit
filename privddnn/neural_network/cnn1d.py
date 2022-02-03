import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import os.path
import h5py
from collections import OrderedDict

from privddnn.classifier import ModelMode, OpName
from .base import NeuralNetwork
from .layers import conv2d, dense, dropout, fitnet_block


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

        filters_one = tf1.get_variable(name='filters-one',
                                      shape=[1, input_shape[2], 1],
                                      dtype=tf2.float32,
                                      trainable=True)
        bias_one = tf1.get_variable(name='bias-one',
                                    shape=[1, 1, 1],
                                    dtype=tf2.float32,
                                    trainable=True)

        conv_one = tf2.nn.conv1d(input=inputs,
                                 filters=filters_one,
                                 stride=1,
                                 padding='SAME',
                                 name='conv-one')

        conv_one = tf2.nn.leaky_relu(conv_one + bias_one, alpha=0.25)

        conv_one_shape = conv_one.get_shape()
        flattened_one = tf2.reshape(conv_one, shape=(-1, np.prod(conv_one_shape[1:])))

        filters_two = tf1.get_variable(name='filters-two',
                                      shape=[1, input_shape[2] + 1, 8],
                                      dtype=tf2.float32,
                                      trainable=True)
        bias_two = tf1.get_variable(name='bias-two',
                                    shape=[1, 1, 8],
                                    dtype=tf2.float32,
                                    trainable=True)

        inputs_concat = tf2.concat([inputs, conv_one], axis=-1)
        conv_two = tf2.nn.conv1d(input=inputs_concat,
                                 filters=filters_two,
                                 stride=1,
                                 padding='SAME',
                                 name='conv-two')

        conv_two = tf2.nn.leaky_relu(conv_two + bias_two, alpha=0.25)  # [B, D, K]

        conv_two_shape = conv_two.get_shape()
        flattened_two = tf2.reshape(conv_two, shape=(-1, np.prod(conv_two_shape[1:])))

        logits_one = dense(inputs=flattened_one,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=True,
                           name='output-one')  # [B, K]

        logits_two = dense(inputs=flattened_two,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=True,
                           name='output-two')  # [B, K]

        logits = tf2.concat([tf2.expand_dims(logits_one, axis=1), tf2.expand_dims(logits_two, axis=1)], axis=1)  # [B, 2, K]
        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        # Tile the labels along an expanded axis
        labels = tf2.expand_dims(labels, axis=1)
        labels = tf2.tile(labels, multiples=(1, self.num_outputs))  # [B, 2]

        # Compute the per-output loss and aggregate for each sample
        sample_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, 2]
        aggregate_sample_loss = tf2.reduce_sum(sample_loss * tf2.constant([0.3, 0.7], dtype=sample_loss.dtype), axis=-1)  # [B]

        return tf2.reduce_mean(aggregate_sample_loss)
