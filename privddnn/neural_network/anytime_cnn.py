import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
from typing import Tuple

from privddnn.utils.constants import SMALL_NUMBER
from .early_exit_dnn import EarlyExitNeuralNetwork
from .layers import conv2d, dense
from .constants import ModelMode


def interleaved_fitnet_block(layer_one_inputs: tf2.Tensor, layer_two_inputs: tf2.Tensor, num_filters: int, pool_size: int, pool_stride: int, name: str) -> Tuple[tf2.Tensor, tf2.Tensor]:
    with tf1.variable_scope(name):

        block_one = conv2d(inputs=layer_one_inputs,
                           filter_size=3,
                           stride=1,
                           num_filters=num_filters,
                           activation='relu',
                           name='block1')

        layer_two_inputs_transformed = 0.5 * (layer_one_inputs + layer_two_inputs)
        block_two = conv2d(inputs=layer_two_inputs_transformed,
                           filter_size=3,
                           stride=1,
                           num_filters=num_filters,
                           activation='relu',
                           name='block2')

        pooled_one = tf2.nn.max_pool(input=block_one,
                                     ksize=pool_size,
                                     strides=[1, pool_stride, pool_stride, 1],
                                     padding='SAME')

        pooled_two = tf2.nn.max_pool(input=block_two,
                                     ksize=pool_size,
                                     strides=[1, pool_stride, pool_stride, 1],
                                     padding='SAME')

        return pooled_one, pooled_two


class AnytimeCNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'anytime-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        is_fine_tune = (model_mode == ModelMode.FINE_TUNE)

        # Create the convolutions
        conv1_layer1, conv1_layer2 = interleaved_fitnet_block(layer_one_inputs=inputs,
                                                              layer_two_inputs=inputs,
                                                              num_filters=16,
                                                              pool_size=4,
                                                              pool_stride=2,
                                                              name='block1')

        conv2_layer1, conv2_layer2 = interleaved_fitnet_block(layer_one_inputs=conv1_layer1,
                                                              layer_two_inputs=conv1_layer2,
                                                              num_filters=16,
                                                              pool_size=4,
                                                              pool_stride=2,
                                                              name='block2')

        conv3_layer1, conv3_layer2 = interleaved_fitnet_block(layer_one_inputs=conv2_layer1,
                                                              layer_two_inputs=conv2_layer2,
                                                              num_filters=12,
                                                              pool_size=2,
                                                              pool_stride=1,
                                                              name='block3')

        # Create the first output layer. We use global average pooling here to reduce the number
        # of trainable parameters
        flattened_one = tf2.reduce_mean(conv3_layer1, axis=[1, 2])  # [B, C]

        output_one = dense(inputs=flattened_one,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output1')  # [B, K]

        # Create the second output layer
        conv3_layer2_shape = conv3_layer2.get_shape()
        flattened_two = tf2.reshape(conv3_layer2, (-1, np.prod(conv3_layer2_shape[1:])))  # [B, H * W * C]

        output_two = dense(inputs=flattened_two,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output2')  # [B, K]

        # Stack the logits together
        output_one = tf2.expand_dims(output_one, axis=1)  # [B, 1, K]
        output_two = tf2.expand_dims(output_two, axis=1)  # [B, 1, K]

        logits = tf2.concat([output_one, output_two], axis=1)  # [B, 2, K]

        if is_fine_tune:
            logits = tf2.stop_gradient(logits)

        self.create_stop_layer(logits=logits)

        return logits
