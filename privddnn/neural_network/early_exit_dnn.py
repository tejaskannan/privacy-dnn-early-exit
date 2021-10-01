import tensorflow as tf2
import tensorflow.compat.v1 as tf1

from privddnn.classifier import OpName, ModelMode
from privddnn.utils.constants import SMALL_NUMBER
from .base import NeuralNetwork
from .constants import STOP_RATES
from .layers import dense
from .layers.layer_utils import differentiable_abs


class EarlyExitNeuralNetwork(NeuralNetwork):

    @property
    def num_outputs(self):
        raise NotImplementedError()

    def create_stop_layer(self, logits: tf2.Tensor):
        probs = tf2.nn.softmax(logits, axis=-1)  # [B, L, K]
        normalized_logits = tf2.math.log(probs + SMALL_NUMBER)  # [B, L, K]
        entropy = tf2.reduce_sum(-1 * probs * normalized_logits, axis=-1, keepdims=True)  # [B, L, 1]

        layer_var = tf1.get_variable(shape=(self.num_outputs, 1),
                                     dtype=tf1.float32,
                                     initializer=tf1.glorot_uniform_initializer(),
                                     trainable=True,
                                     name='layer-var')

        layer_var = tf2.expand_dims(layer_var, axis=0)  # [1, L, 1]
        layer_var = tf2.tile(layer_var, multiples=(tf2.shape(entropy)[0], 1, 1))  # [B, L, 1]

        stop_features = tf2.concat([probs, entropy, layer_var], axis=-1)  # [B, L, K + 2]
        stop_state = dense(inputs=stop_features,
                           output_units=1,
                           use_dropout=False,
                           dropout_keep_rate=1.0,
                           activation='linear',
                           name='stop')

        stop_probs = tf2.math.sigmoid(tf2.squeeze(stop_state, axis=-1))  # [B, L]

        # Scale the stop probs by the continue rates from previous layers
        continue_probs = tf2.math.cumprod(1.0 - stop_probs, axis=-1, exclusive=True)
        stop_probs = continue_probs * stop_probs

        # Default to the last layer rate
        stop_probs = stop_probs[:, 0:-1]  # [B, L - 1]
        total_probs = tf2.reduce_sum(stop_probs, axis=-1, keepdims=True)  # [B, 1]
        last_prob = tf2.nn.relu(1.0 - total_probs)

        stop_probs = tf2.concat([stop_probs, last_prob], axis=-1)  # [B, L]

        self._ops[OpName.STOP_PROBS] = stop_probs

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        labels = tf2.expand_dims(labels, axis=1)  # [B, 1]
        labels = tf2.tile(labels, multiples=(1, self.num_outputs))  # [B, L]

        pred_loss = tf2.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B, L]

        if model_mode == ModelMode.FINE_TUNE:
            # Weight the prediction loss by the stop probabilities (want high prob on low cross entropy)
            stop_probs = self._ops[OpName.STOP_PROBS]
            preds = tf2.cast(tf2.argmax(logits, axis=-1), dtype=labels.dtype)  # [B, L]
            is_incorrect = 1.0 - tf2.cast(tf2.equal(preds, labels), dtype=stop_probs.dtype)

            sample_loss = tf2.reduce_sum(is_incorrect * stop_probs, axis=-1)  # [B]

            #sample_loss = tf2.reduce_sum(pred_loss * stop_probs, axis=-1)  # [B]

            # Get the loss required to meet the target rates
            target_rates = tf2.constant(self.hypers[STOP_RATES])
            target_rates = tf2.expand_dims(target_rates, axis=0)  # [1, L]
            rate_loss = differentiable_abs(stop_probs - target_rates)  # [B, L]
            rate_loss = tf2.reduce_sum(rate_loss, axis=-1)  # [B]

            # Combine the loss terms
            sample_loss = tf2.add(sample_loss, 10.0 * rate_loss)
        else:
            weights = tf2.constant([[0.3, 0.7]], dtype=pred_loss.dtype)
            pred_loss = tf2.math.multiply(pred_loss, weights)
            sample_loss = tf2.reduce_sum(pred_loss, axis=-1)

        return tf2.reduce_mean(sample_loss)  # Scalar
