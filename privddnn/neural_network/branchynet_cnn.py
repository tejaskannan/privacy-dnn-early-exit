import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from .base import NeuralNetwork
from .layers import conv2d, dense, bottleneck_block
from .constants import OpName


class BranchyNetCNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-cnn'

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int) -> tf2.Tensor:
        # Create the convolutions
        block_one = conv2d(inputs=inputs,
                           filter_size=5,
                           stride=2,
                           num_filters=16,
                           activation='relu',
                           name='block1')

        block_two = conv2d(inputs=inputs,
                           filter_size=3,
                           stride=1,
                           num_filters=24,
                           activation='relu',
                           name='block2')

        block_three = conv2d(inputs=inputs,
                             filter_size=3,
                             stride=1,
                             num_filters=32,
                             activation='relu',
                             name='block3')

        # Create the first output layer
        pooled_one = tf2.nn.avg_pool(input=block_one,
                                     ksize=3,
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        pooled_one_shape = pooled_one.get_shape()
        flattened_one = tf2.reshape(pooled_one, (-1, np.prod(pooled_one_shape[1:])))  # [B, D]
        output_one = dense(inputs=flattened_one,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output1')  # [B, K]

        # Create the second output layer
        pooled_two = tf2.nn.avg_pool(input=block_three,
                                     ksize=3,
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        pooled_two_shape = pooled_two.get_shape()
        flattened_two = tf2.reshape(pooled_two, (-1, np.prod(pooled_two_shape[1:])))

        hidden = dense(inputs=flattened_two,
                       output_units=128,
                       use_dropout=False,
                       dropout_keep_rate=dropout_keep_rate,
                       activation='relu',
                       name='output2-hidden')

        output_two = dense(inputs=hidden,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output2')  # [B, K]

        # Stack the logits together
        output_one = tf2.expand_dims(output_one, axis=1)  # [B, 1, K]
        output_two = tf2.expand_dims(output_two, axis=1)  # [B, 1, K]

        logits = tf2.concat([output_one, output_two], axis=1)  # [B, 2, K]

        self._ops[OpName.STATE] = flattened_one

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder) -> tf2.Tensor:
        num_outputs = logits.get_shape()[1]  # L
        labels = tf2.expand_dims(labels, axis=1)  # [B, 1]
        labels = tf2.tile(labels, multiples=(1, num_outputs))  # [B, L]

        loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]
        loss_weights = tf2.constant([[0.5, 0.5]], dtype=loss.dtype)

        weighted_loss = tf2.reduce_sum(loss * loss_weights, axis=-1)  # [B]
        return tf2.reduce_mean(weighted_loss)  # Scalar
