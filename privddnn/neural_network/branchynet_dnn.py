import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Concatenate
from typing import List

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE
from .early_exit_dnn import EarlyExitNeuralNetwork


class BranchyNetDNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(16, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(128, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout3)

        return [output0, output1]



class BranchyNetDNNMinimized(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-min'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(8, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(32, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(32, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout3)

        return [output0, output1]



class BranchyNetDNN3(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-3'

    @property
    def num_outputs(self) -> int:
        return 3

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(16, activation='relu')(inputs)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(64, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(128, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout1)
        output2 = Dense(num_labels, activation='softmax', name='output2')(dropout3)

        return [output0, output1, output2]


class BranchyNetDNN4(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-4'

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

        concat = Concatenate(axis=-1)([inputs, dropout0])
        hidden1 = Dense(64, activation='relu')(concat)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        hidden2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(128, activation='relu')(dropout2)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout0)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout1)
        output2 = Dense(num_labels, activation='softmax', name='output2')(dropout2)
        output3 = Dense(num_labels, activation='softmax', name='output3')(dropout3)

        return [output0, output1, output2, output3]


class BranchyNetDNNAlt(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-dnn-alt'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        if len(inputs.get_shape()) > 2:
            inputs = tf.keras.backend.reshape(inputs, shape=(-1, np.prod(inputs.get_shape()[1:])))

        hidden0 = Dense(8, activation='linear')(inputs)
        hidden0 = LeakyReLU(alpha=0.25)(hidden0)
        dropout0 = Dropout(rate=1.0 - dropout_keep_rate)(hidden0, training=is_train)

        hidden1 = Dense(12, activation='linear')(dropout0)
        hidden1 = LeakyReLU(alpha=0.25)(hidden1)
        dropout1 = Dropout(rate=1.0 - dropout_keep_rate)(hidden1, training=is_train)

        concat = Concatenate(axis=-1)([inputs, dropout1])
        hidden2 = Dense(48, activation='linear')(concat)
        hidden2 = LeakyReLU(alpha=0.25)(hidden2)
        dropout2 = Dropout(rate=1.0 - dropout_keep_rate)(hidden2, training=is_train)

        hidden3 = Dense(48, activation='linear')(dropout2)
        hidden3 = LeakyReLU(alpha=0.25)(hidden3)
        dropout3 = Dropout(rate=1.0 - dropout_keep_rate)(hidden3, training=is_train)

        hidden4 = Dense(48, activation='linear')(dropout3)
        hidden4 = LeakyReLU(alpha=0.25)(hidden4)
        dropout4 = Dropout(rate=1.0 - dropout_keep_rate)(hidden4, training=is_train)

        output0 = Dense(num_labels, activation='softmax', name='output0')(dropout1)
        output1 = Dense(num_labels, activation='softmax', name='output1')(dropout4)

        return [output0, output1]
