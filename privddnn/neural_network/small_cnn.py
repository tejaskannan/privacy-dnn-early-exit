import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from .base import NeuralNetwork
from .layers import conv2d, dense


class SmallCNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'small-cnn'

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int) -> tf2.Tensor:
        # Apply the convolution layers
        conv_one = conv2d(inputs=inputs,
                          filter_size=5,
                          stride=2,
                          num_filters=16,
                          activation='relu',
                          name='conv1')

        # Average pool the result
        pooled = tf2.nn.avg_pool(input=conv_one,
                                 ksize=3,
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')

        # Create the output layer
        pooled_shape = pooled.get_shape()
        flattened = tf2.reshape(pooled, (-1, np.prod(pooled_shape[1:])))

        logits = dense(inputs=flattened,
                        output_units=num_labels,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='linear',
                        name='output')  # [B, K]

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder) -> tf2.Tensor:
        loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]
        return tf2.reduce_mean(loss)
