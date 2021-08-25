import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from privddnn.utils.constants import SMALL_NUMBER
from .base import NeuralNetwork
from .layers import conv2d, dense, bottleneck_block
from .layers.layer_utils import differentiable_abs
from .constants import OpName, ModelMode, STOP_RATES


class BranchyNetCNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-cnn'

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

        # Create the stop layer
        probs = tf2.nn.softmax(logits, axis=-1)  # [B, 2, K]
        normalized_logits = tf2.math.log(probs + SMALL_NUMBER)  # [B, 2, K]
        entropy = tf2.reduce_sum(-1 * probs * normalized_logits, axis=-1, keepdims=True)  # [B, 2, 1]

        layer_var = tf1.get_variable(shape=(2, 1),
                                     dtype=tf1.float32,
                                     initializer=tf1.glorot_uniform_initializer(),
                                     trainable=True,
                                     name='layer-var')

        layer_var = tf2.expand_dims(layer_var, axis=0)  # [1, 2, 1]
        layer_var = tf2.tile(layer_var, multiples=(tf2.shape(entropy)[0], 1, 1))  # [B, 2, 1]

        stop_features = tf2.concat([probs, entropy, layer_var], axis=-1)  # [B, 2, K + 2]
        stop_state = dense(inputs=stop_features,
                           output_units=1,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='stop')

        stop_probs = tf2.math.sigmoid(tf2.squeeze(stop_state, axis=-1))  # [B, 2]
        stop_probs = tf2.expand_dims(stop_probs[:, 0], axis=-1)
        stop_probs = tf2.concat([stop_probs, (1.0 - stop_probs)], axis=-1)

        #continue_probs = tf2.math.cumprod(1.0 - stop_probs, exclusive=True)
        #stop_probs = stop_probs * continue_probs

        self._ops[OpName.STOP_PROBS] = stop_probs

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        num_outputs = logits.get_shape()[1]  # L
        labels = tf2.expand_dims(labels, axis=1)  # [B, 1]
        labels = tf2.tile(labels, multiples=(1, num_outputs))  # [B, L]

        pred_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]

        if model_mode == ModelMode.FINE_TUNE:
            # Weight the prediction loss by the stop probabilities (want high prob on low cross entropy)
            stop_probs = self._ops[OpName.STOP_PROBS]
            sample_loss = tf2.reduce_sum(pred_loss * stop_probs, axis=-1)  # [B]

            # Get the loss required to meet the target rates
            target_rates = tf2.constant(self.hypers[STOP_RATES])
            target_rates = tf2.expand_dims(target_rates, axis=0)  # [1, L]
            rate_loss = differentiable_abs(stop_probs - target_rates)  # [B, L]
            rate_loss = tf2.reduce_sum(rate_loss, axis=-1)  # [B]

            # Combine the loss terms
            sample_loss = tf2.add(sample_loss, 10.0 * rate_loss)
        else:
            sample_loss = tf2.reduce_mean(pred_loss, axis=-1)

        return tf2.reduce_mean(sample_loss)  # Scalar
