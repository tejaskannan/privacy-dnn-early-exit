import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, GRU, Layer, Dropout
from typing import List

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE
from .early_exit_dnn import EarlyExitNeuralNetwork


class BranchyNetRNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-rnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]

        if len(inputs.get_shape()) == 4:
            inputs = tf.keras.backend.squeeze(inputs, axis=-1)

        rnn0 = GRU(units=128, return_sequences=True)(inputs)
        rnn1 = GRU(units=128, return_sequences=True)(rnn0)
        rnn2 = GRU(units=128, return_sequences=True)(rnn1)

        rnn0_state = rnn0[:, -1, :]  # [B, D]
        rnn2_state = rnn2[:, -1, :]  # [B, D]

        output0 = Dense(num_labels, activation='softmax', name='output0')(rnn0_state)

        output1_hidden = Dense(64, activation='relu')(rnn2_state)
        output1_dropout = Dropout(rate=1.0 - dropout_keep_rate)(output1_hidden)
        output1 = Dense(num_labels, activation='softmax', name='output1')(output1_dropout)

        return [output0, output1]

