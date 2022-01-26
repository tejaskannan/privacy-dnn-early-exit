import tensorflow as tf2
import tensorflow.compat.v1 as tf1

from privddnn.utils.constants import SMALL_NUMBER
from .base import NeuralNetwork
from .constants import OpName, ModelMode, STOP_RATES
from .layers import dense
from .layers.layer_utils import differentiable_abs


class StopModel(NeuralNetwork):

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
        # Weight the prediction loss by the stop probabilities (want high prob on low cross entropy)
        stop_probs = self._ops[OpName.STOP_PROBS]

        # Get the loss required to meet the target rates
        observed_rates = compute_average_rates_per_label(rates=stop_probs,
                                                         labels=labels)

        sample_loss = differentiable_abs(stop_probs - target_rates)  # [B]
        return tf2.reduce_mean(sample_loss)  # Scalar
