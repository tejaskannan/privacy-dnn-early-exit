import tensorflow as tf2
import tensorflow.compat.v1 as tf1

from privddnn.utils.tf_utils import compute_average_rates_per_label
from .stop_model import StopNeuralNetwork
from .layers import dense
from .constants import PhName


class StopDNN(StopNeuralNetwork):

    @property
    def name(self) -> str:
        return 'stop-dnn'

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf2.Tensor, num_labels: int) -> tf2.Tensor:
        hidden = dense(inputs=inputs,
                       output_units=16,
                       use_dropout=False,
                       dropout_keep_rate=dropout_keep_rate,
                       activation='relu',
                       name='hidden')

        # TODO: Get the number of outputs in a more general way. Do this via load_metadata() in StopNeuralNetwork

        logits = dense(inputs=hidden,
                        output_units=2,
                        use_dropout=False,
                        dropout_keep_rate=dropout_keep_rate,
                        activation='linear',
                        name='output')  # [B, K]

        return logits

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder) -> tf2.Tensor:
        stop_rates = tf2.nn.softmax(logits, axis=-1)  # [B, K]
        avg_rates = compute_average_rates_per_label(rates=stop_rates, labels=labels)  # [L, K]

        model_incorrect = (1.0 - self._placeholders[PhName.MODEL_CORRECT])  # [B, K]

        pred_loss = tf2.reduce_mean(stop_rates * model_incorrect, axis=0)  # [K]

        target_rates = tf2.expand_dims(self.placeholders[PhName.STOP_RATES], axis=0)  # [1, K]
        stop_loss = tf2.reduce_mean(tf2.square(avg_rates - target_rates), axis=0)  # [K]

        print(pred_loss)
        print(stop_loss)

        loss = pred_loss + self._placeholders[PhName.LOSS_WEIGHT] * stop_loss
        print(loss)

        return tf2.reduce_sum(loss)
