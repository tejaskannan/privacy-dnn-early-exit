import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from .base import NeuralNetwork
from .layers import conv2d, dense
from .constants import MetaName, ModelMode


class BranchyNetDNN(NeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn'

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        # Flatten the inputs
        num_features = np.prod(self.metadata[MetaName.INPUT_SHAPE])
        inputs = tf2.reshape(inputs, (-1, num_features))

        # Create the hidden layers
        hidden1 = dense(inputs=inputs,
                        output_units=128,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='leaky_relu',
                        name='hidden1')

        hidden2 = dense(inputs=hidden1,
                        output_units=128,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='leaky_relu',
                        name='hidden2')

        hidden3 = dense(inputs=hidden2,
                        output_units=64,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='leaky_relu',
                        name='hidden3')

        # Create the output layers
        output_one = dense(inputs=hidden1,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output1')  # [B, K]

        output_two = dense(inputs=hidden3,
                           output_units=num_labels,
                           use_dropout=False,
                           dropout_keep_rate=dropout_keep_rate,
                           activation='linear',
                           name='output2')  # [B, K]

        # Stack the logits together
        output_one = tf2.expand_dims(output_one, axis=1)  # [B, 1, K]
        output_two = tf2.expand_dims(output_two, axis=1)  # [B, 1, K]

        logits = tf2.concat([output_one, output_two], axis=1)  # [B, 2, K]

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        num_outputs = logits.get_shape()[1]  # L
        labels = tf2.expand_dims(labels, axis=1)  # [B, 1]
        labels = tf2.tile(labels, multiples=(1, num_outputs))  # [B, L]

        loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]
        loss_weights = tf2.constant([[0.3, 0.7]], dtype=loss.dtype)

        weighted_loss = tf2.reduce_sum(loss * loss_weights, axis=-1)  # [B]
        return tf2.reduce_mean(weighted_loss)  # Scalar
