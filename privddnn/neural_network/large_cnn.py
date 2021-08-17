import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from .base import NeuralNetwork
from .layers import conv2d, dense


class LargeCNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'large-cnn'

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int) -> tf2.Tensor:
        # Apply the convolution layers
        conv_one = conv2d(inputs=inputs,
                          filter_size=5,
                          stride=2,
                          num_filters=16,
                          activation='relu',
                          name='conv1')

        conv_two = conv2d(inputs=conv_one,
                          filter_size=3,
                          stride=1,
                          num_filters=24,
                          activation='relu',
                          name='conv2')

        conv_three = conv2d(inputs=conv_two,
                            filter_size=3,
                            stride=1,
                            num_filters=32,
                            activation='relu',
                            name='conv3')

        # Average pool the result
        pooled = tf2.nn.avg_pool(input=conv_three,
                                 ksize=3,
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')

        # Create the output layer
        pooled_shape = pooled.get_shape()
        flattened = tf2.reshape(pooled, (-1, np.prod(pooled_shape[1:])))

        hidden = dense(inputs=flattened,
                       output_units=64,
                       use_dropout=False,
                       dropout_keep_rate=dropout_keep_rate,
                       activation='relu',
                       name='output-hidden')

        logits = dense(inputs=hidden,
                        output_units=num_labels,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='linear',
                        name='output')  # [B, K]

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder) -> tf2.Tensor:
        loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]
        return tf2.reduce_mean(loss)
