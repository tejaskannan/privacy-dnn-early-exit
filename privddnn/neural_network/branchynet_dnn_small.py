import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.layers import BatchNormalization, Concatenate
from typing import List, Dict

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE
from .early_exit_dnn import EarlyExitNeuralNetwork


class BranchyNetDNNSmall(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-small'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.1,
            'output1': 0.9
        }

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(32, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(32, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(64, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout3)

        return [output0, output1]


class BranchyNetDNNSmall3(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-small-3'

    @property
    def num_outputs(self) -> int:
        return 3

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.2,
            'output1': 0.6,
            'output2': 1.0
        }

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(32, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(32, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(64, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout1)
        output2 = Dense(num_labels, activation='softmax', name='output2')(dropout3)

        return [output0, output1, output2]


class BranchyNetDNNSmall4(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-small-4'

    @property
    def num_outputs(self) -> int:
        return 4

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(16, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        hidden1 = Dense(16, activation='relu')(dropout0)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(32, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout1)
        output2 = Dense(num_labels, activation='softmax', name='output2')(dropout2)
        output3 = Dense(num_labels, activation='softmax', name='output3')(dropout3)

        return [output0, output1, output2, output3]


class BranchyNetDNNSmallAlt(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-small-alt'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.1,
            'output1': 0.9
        }

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(64, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        hidden1 = Dense(128, activation='relu')(dropout0)
        hidden2 = Dense(128, activation='relu')(hidden1)
        hidden3 = Dense(128, activation='relu')(hidden2)

        output0 = Dense(num_labels, activation='softmax', name='output0')(hidden0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(hidden3)

        return [output0, output1]

