import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from privddnn.classifier import ModelMode
from .early_exit_dnn import EarlyExitNeuralNetwork
from .layers import conv2d, dense, dropout, fitnet_block


class BranchyNetCNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        is_fine_tune = (model_mode == ModelMode.FINE_TUNE)
        is_train = (model_mode == ModelMode.TRAIN)

        if is_fine_tune:
            inputs = self.perturb_inputs(inputs=inputs)

        # Create the convolution blocks
        conv1_block, _, _ = fitnet_block(inputs=inputs, num_filters=16, pool_size=4, pool_stride=2, trainable=is_train, name='block1')
        conv2_block, conv2_interm, _ = fitnet_block(inputs=conv1_block, num_filters=16, pool_size=4, pool_stride=2, trainable=is_train, name='block2')
        conv3_block, _, _ = fitnet_block(inputs=conv2_block, num_filters=12, pool_size=2, pool_stride=1, trainable=is_train, name='block3')

        # Create the first output layer. We use global average pooling here to reduce the number of parameters
        flattened_one = tf2.reduce_mean(conv2_interm, axis=[1, 2])  # [B, C]
        output_one = dense(inputs=flattened_one,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=is_train,
                           name='output1')  # [B, K]

        # Create the second output layer by first flattening out the pixels
        conv3_block_shape = conv3_block.get_shape()
        flattened_two = tf2.reshape(conv3_block, (-1, np.prod(conv3_block_shape[1:])))

        output_two = dense(inputs=flattened_two,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=is_train,
                           name='output2')  # [B, K]

        # Stack the logits together
        output_one = tf2.expand_dims(output_one, axis=1)  # [B, 1, K]
        output_two = tf2.expand_dims(output_two, axis=1)  # [B, 1, K]

        logits = tf2.concat([output_one, output_two], axis=1)  # [B, 2, K]
        return logits


class BranchyNetCNNSmall(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-cnn-small'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        is_fine_tune = (model_mode == ModelMode.FINE_TUNE)

        # Create the convolution blocks
        conv1_block, _, _ = fitnet_block(inputs=inputs, num_filters=8, pool_size=4, pool_stride=2, name='block1', trainable=True)
        #conv2_block, conv2_interm, _ = fitnet_block(inputs=conv1_block, num_filters=6, pool_size=4, pool_stride=2, name='block2')
        conv2_block, _, _ = fitnet_block(inputs=conv1_block, num_filters=8, pool_size=2, pool_stride=1, name='block2', trainable=True)

        # Create the first output layer. We use global average pooling here to reduce the number of parameters
        flattened_one = tf2.reduce_mean(conv1_block, axis=[1, 2])  # [B, C]
        output_one = dense(inputs=flattened_one,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=True,
                           name='output1')  # [B, K]

        # Create the second output layer by first flattening out the pixels
        conv2_block_shape = conv2_block.get_shape()
        flattened_two = tf2.reshape(conv2_block, (-1, np.prod(conv2_block_shape[1:])))

        output_two = dense(inputs=flattened_two,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           trainable=True,
                           name='output2')  # [B, K]

        # Stack the logits together
        output_one = tf2.expand_dims(output_one, axis=1)  # [B, 1, K]
        output_two = tf2.expand_dims(output_two, axis=1)  # [B, 1, K]

        logits = tf2.concat([output_one, output_two], axis=1)  # [B, 2, K]

        if is_fine_tune:
            logits = tf2.stop_gradient(logits)

        #self.create_stop_layer(logits=logits)

        return logits
