import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from .constants import ModelMode
from .early_exit_dnn import EarlyExitNeuralNetwork
from .layers import conv2d, dense, dropout


class IndependentCNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'independent-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        is_fine_tune = (model_mode == ModelMode.FINE_TUNE)

        # Create the convolutions
        block_one = conv2d(inputs=inputs,
                           filter_size=5,
                           stride=2,
                           num_filters=16,
                           activation='relu',
                           name='block1')

        block_two = conv2d(inputs=inputs,
                           filter_size=5,
                           stride=2,
                           num_filters=16,
                           activation='relu',
                           name='block2')

        block_three = conv2d(inputs=block_two,
                             filter_size=3,
                             stride=1,
                             num_filters=32,
                             activation='relu',
                             name='block3')

        # Create the first output layer
        pooled_one = tf2.nn.avg_pool(input=block_one,
                                     ksize=5,
                                     strides=[1, 4, 4, 1],
                                     padding='SAME')

        pooled_one_shape = pooled_one.get_shape()
        flattened_one = tf2.reshape(pooled_one, (-1, np.prod(pooled_one_shape[1:])))  # [B, D]

        if is_fine_tune:
            flattened_one = tf2.stop_gradient(flattened_one)

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

        if is_fine_tune:
            hidden = tf2.stop_gradient(hidden)

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

        self.create_stop_layer(logits=logits)

        return logits
