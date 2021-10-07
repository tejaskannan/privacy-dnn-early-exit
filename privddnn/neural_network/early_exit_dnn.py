import tensorflow as tf2
import tensorflow.compat.v1 as tf1

from privddnn.classifier import OpName, ModelMode
from privddnn.utils.tf_utils import tf_compute_entropy, make_max_prob_targets
from privddnn.utils.constants import SMALL_NUMBER
from .base import NeuralNetwork
from .constants import STOP_RATES
from .layers import dense, conv2d
from .layers.layer_utils import differentiable_abs


class EarlyExitNeuralNetwork(NeuralNetwork):

    @property
    def num_outputs(self):
        raise NotImplementedError()

    def perturb_inputs(self, inputs: tf2.Tensor) -> tf2.Tensor:
        input_shape = inputs.get_shape()
        if len(input_shape) == 3:
            inputs = tf2.expand_dims(inputs, axis=-1)

        weights = tf1.get_variable(name='perturb',
                                   shape=inputs.get_shape()[1:],
                                   initializer=tf1.glorot_uniform_initializer(),
                                   trainable=True)

        #hidden = conv2d(inputs=inputs,
        #                num_filters=4,
        #                filter_size=3,
        #                stride=1,
        #                activation='relu',
        #                trainable=True,
        #                name='perturb-hidden')

        #transformed = conv2d(inputs=inputs,
        #                     num_filters=inputs.get_shape()[-1],
        #                     filter_size=3,
        #                     stride=1,
        #                     activation='linear',
        #                     trainable=True,
        #                     name='perturb')

        perturbation = tf2.expand_dims(0.5 * tf2.nn.tanh(weights), axis=0)
        result = inputs + perturbation

        return result


    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        if model_mode == ModelMode.FINE_TUNE:
            # Weight the prediction loss by the stop probabilities (want high prob on low cross entropy)
            probs = tf2.nn.softmax(logits, axis=-1)
            max_probs = tf2.reduce_max(probs, axis=-1)  # [B, L]

            avg_max_prob = tf2.reduce_mean(max_probs, axis=0, keepdims=True)  # [1, L]
            adjusted_loss = differentiable_abs(max_probs - avg_max_prob)  # [B, L]

            #thresholds = tf2.constant([0.87514, 0.99561])
            #thresholds = tf2.expand_dims(thresholds, axis=0)  # [1, L]

            #max_probs = tf2.reduce_max(probs, axis=-1)  # [B, L]
            #adjusted_loss = differentiable_abs(max_probs - thresholds)

            #num_labels = probs.get_shape()[-1]
            #target_dist = make_max_prob_targets(labels=labels, num_labels=num_labels, target_prob=thresholds) # [B, L, K]

            #adjusted_loss = tf2.nn.softmax_cross_entropy_with_logits(labels=target_dist, logits=logits, axis=-1)  # [B, L]
            sample_loss = tf2.reduce_sum(adjusted_loss, axis=-1)  # [B]
        else:
            labels = tf2.expand_dims(labels, axis=1)  # [B, 1]
            labels = tf2.tile(labels, multiples=(1, self.num_outputs))  # [B, L]

            pred_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]

            weights = tf2.constant([[0.3, 0.7]], dtype=pred_loss.dtype)
            pred_loss = tf2.math.multiply(pred_loss, weights)
            sample_loss = tf2.reduce_sum(pred_loss, axis=-1)

        return tf2.reduce_mean(sample_loss)  # Scalar
