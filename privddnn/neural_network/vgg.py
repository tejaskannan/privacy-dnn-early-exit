"""
Pretrianed VGG model from: https://github.com/geifmany/cifar-vgg/
"""
import numpy as np
import os.path
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Layer, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import Metric
from typing import List, Dict

from privddnn.classifier import ModelMode
from .early_exit_dnn import EarlyExitNeuralNetwork


class VGG(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'vgg'

    @property
    def num_outputs(self) -> int:
        return 2

    def make_loss(self) -> Dict[str, str]:
        return {
            'output0': 'sparse_categorical_crossentropy',
            'dense_2': 'sparse_categorical_crossentropy'
        }

    def make_metrics(self) -> Dict[str, Metric]:
        return {
            'output0': tf.metrics.SparseCategoricalAccuracy(name='acc'),
            'dense_2': tf.metrics.SparseCategoricalAccuracy(name='acc')
        }

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.25,
            'dense_2': 0.75
        }

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TRAIN else self.hypers[DROPOUT_KEEP_RATE]

        conv0 = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_1')(inputs)
        batchnorm0 = BatchNormalization(name='batch_normalization_1')(conv0)

        conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_2')(batchnorm0)
        batchnorm1 = BatchNormalization(name='batch_normalization_2')(conv1)
        pooled1 = MaxPooling2D(pool_size=(2, 2))(batchnorm1)

        conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_3')(pooled1)
        batchnorm2 = BatchNormalization(name='batch_normalization_3')(conv2)

        conv3 = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_4')(batchnorm2)
        batchnorm3 = BatchNormalization(name='batch_normalization_4')(conv3)
        pooled3 = MaxPooling2D(pool_size=(2, 2))(batchnorm3)

        conv4 = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_5')(pooled3)
        batchnorm4 = BatchNormalization(name='batch_normalization_5')(conv4)

        conv5 = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_6')(batchnorm4)
        batchnorm5 = BatchNormalization(name='batch_normalization_6')(conv5)

        conv6 = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_7')(batchnorm5)
        batchnorm6 = BatchNormalization(name='batch_normalization_7')(conv6)
        pooled6 = MaxPooling2D(pool_size=(2, 2))(batchnorm6)

        conv7 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_8')(pooled6)
        batchnorm7 = BatchNormalization(name='batch_normalization_8')(conv7)

        conv8 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_9')(batchnorm7)
        batchnorm8 = BatchNormalization(name='batch_normalization_9')(conv8)

        conv9 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_10')(batchnorm8)
        batchnorm9 = BatchNormalization(name='batch_normalization_10')(conv9)
        pooled9 = MaxPooling2D(pool_size=(2, 2))(batchnorm9)

        conv10 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_11')(pooled9)
        batchnorm10 = BatchNormalization(name='batch_normalization_11')(conv10)

        conv11 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_12')(batchnorm10)
        batchnorm11 = BatchNormalization(name='batch_normalization_12')(conv11)

        conv12 = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False, name='conv2d_13')(batchnorm11)
        batchnorm12 = BatchNormalization(name='batch_normalization_13')(conv12)
        pooled12 = MaxPooling2D(pool_size=(2, 2))(batchnorm12)

        # Create the output layers
        flattened0 = Flatten()(pooled1)
        output0 = Dense(num_labels, activation='softmax', trainable=True, name='output0')(flattened0)

        flattened1 = Flatten()(pooled12)
        hidden1 = Dense(512, activation='relu', trainable=False, name='dense_1')(flattened1)
        hidden1_batchnorm = BatchNormalization(name='batch_normalization_14')(hidden1)
        output1 = Dense(num_labels, activation='softmax', trainable=False, name='dense_2')(hidden1_batchnorm)

        return [output0, output1]

    def make(self, model_mode: ModelMode):
        super().make(model_mode=model_mode)

        dir_name = os.path.dirname(__file__)
        self._model.load_weights(os.path.join(dir_name, 'pretrained/cifar10vgg.h5'), by_name=True)
