import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Dropout, Layer, BatchNormalization
from typing import List

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE
from .early_exit_dnn import EarlyExitNeuralNetwork


class BranchyNetCNN(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'branchynet-cnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]

        conv0 = Conv2D(filters=8, kernel_size=3, activation='relu')(inputs)
        batchnorm0 = BatchNormalization()(conv0)
        pooled0 = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(batchnorm0)

        conv1 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(pooled0)
        batchnorm1 = BatchNormalization()(conv1)

        conv2 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(batchnorm1)
        batchnorm2 = BatchNormalization()(conv2)

        pooled2 = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(batchnorm2)

        flattened0 = Flatten()(pooled0)
        output0 = Dense(10, activation='softmax', name='output0')(flattened0)

        flattened2 = Flatten()(pooled2)
        output1_hidden = Dense(128, activation='relu')(flattened2)
        output1 = Dense(10, activation='softmax', name='output1')(output1_hidden)

        return [output0, output1]

