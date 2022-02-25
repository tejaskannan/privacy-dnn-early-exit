import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Dropout, Layer
from tensorflow.keras.layers import BatchNormalization
from typing import List

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE
from .early_exit_dnn import EarlyExitNeuralNetwork


class SpeechCNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'speech-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]
        is_train = (model_mode == ModelMode.TRAIN)

        downsampled = tf.keras.layers.experimental.preprocessing.Resizing(32, 32)(inputs)

        conv0 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), activation='relu')(downsampled)
        batchnorm0 = BatchNormalization()(conv0)

        conv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), activation='relu')(batchnorm0)
        batchnorm1 = BatchNormalization()(conv1)
        pooled1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnorm1)

        conv2 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(pooled1)
        batchnorm2 = BatchNormalization()(conv2)

        conv3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(batchnorm2)
        batchnorm3 = BatchNormalization()(conv3)

        conv4 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), activation='relu')(batchnorm3)
        batchnorm4 = BatchNormalization()(conv4)
        pooled4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnorm4)

        output0_pooled = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(batchnorm1)
        flattened0 = Flatten()(output0_pooled)
        output0_hidden = Dense(64, activation='relu')(flattened0)
        output0_dropout = Dropout(rate=1.0 - dropout_keep_rate)(output0_hidden, training=is_train)
        output0 = Dense(num_labels, activation='softmax', name='output0')(output0_dropout)

        flattened2 = Flatten()(pooled4)
        output1_hidden = Dense(128, activation='relu')(flattened2)
        output1_dropout = Dropout(rate=1.0 - dropout_keep_rate)(output1_hidden, training=is_train)
        output1 = Dense(num_labels, activation='softmax', name='output1')(output1_dropout)

        return [output0, output1]

