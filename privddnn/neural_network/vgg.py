"""
Pretrianed VGG model from: https://github.com/geifmany/cifar-vgg/
"""
import numpy as np
import os.path
import tensorflow as tf
from enum import Enum, auto
from tensorflow.keras.layers import Dense, Dropout, Flatten, Layer, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import Metric
from typing import List, Dict, Any

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE, BATCH_SIZE
from .early_exit_dnn import EarlyExitNeuralNetwork


class CifarMode(Enum):
    CIFAR_10 = auto()
    CIFAR_100 = auto()


class VGG(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'vgg'

    @property
    def default_hypers(self) -> Dict[str, Any]:
        default_hypers = super().default_hypers
        default_hypers['num_outputs'] = 2
        return default_hypers

    @property
    def num_outputs(self) -> int:
        return self.hypers['num_outputs']

    def make_loss(self) -> Dict[str, str]:
        loss_dict: Dict[str, str] = dict()

        for idx in range(self.num_outputs - 1):
            loss_dict['output{}'.format(idx)] = 'sparse_categorical_crossentropy'

        return loss_dict

    def make_metrics(self) -> Dict[str, Metric]:
        metrics_dict: Dict[str, Metric] = dict()

        for idx in range(self.num_outputs - 1):
            metrics_dict['output{}'.format(idx)] = tf.metrics.SparseCategoricalAccuracy('acc')

        return metrics_dict

    def make_loss_weights(self) -> Dict[str, float]:
        weights_dict: Dict[str, float] = dict()

        for idx in range(self.num_outputs - 1):
            weights_dict['output{}'.format(idx)] = 1.0

        return weights_dict

    def compute_probs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the predicted probabilites on the given dataset.
        """
        preds = self._model.predict(inputs, batch_size=self.hypers[BATCH_SIZE], verbose=0)

        if self.num_outputs == 2:
            preds_list = [preds[0], preds[3]]
        elif self.num_outputs == 3:
            preds_list = [preds[0], preds[1], preds[3]]
        elif self.num_outputs == 4:
            preds_list = preds
        else:
            raise ValueError('Can only support a number of outputs in [2, 4]')

        return np.concatenate([np.expand_dims(arr, axis=1) for arr in preds_list])

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_keep_rate = 1.0 if model_mode == ModelMode.TEST else self.hypers[DROPOUT_KEEP_RATE]

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
        outputs: List[Layer] = []

        output0_pooled = pooled1 if self._cifar_mode == CifarMode.CIFAR_10 else pooled3
        flattened0 = Flatten()(output0_pooled)
        output0_hidden = Dense(64, activation='relu', trainable=True, name='output0_hidden0')(flattened0)
        output0_batchnorm = BatchNormalization(name='batch_normalization_output0')(output0_hidden)
        output0 = Dense(num_labels, activation='softmax', trainable=True, name='output0')(output0_batchnorm)

        outputs.append(output0)

        if self.num_outputs >= 3:
            output1_pooled = pooled3 if self._cifar_mode == CifarMode.CIFAR_10 else pooled6
            flattened1 = Flatten()(output1_pooled)
            output1_hidden = Dense(64, activation='relu', trainable=True, name='output1_hidden0')(flattened1)
            output1_batchnorm = BatchNormalization(name='batch_normalization_output1')(output1_hidden)
            output1 = Dense(num_labels, activation='softmax', trainable=True, name='output1')(output1_batchnorm)

            outputs.append(output1)

        if self.num_outputs == 4:
            output2_pooled = pooled6 if self._cifar_mode == CifarMode.CIFAR_10 else pooled9
            flattened2 = Flatten()(output2_pooled)
            output2_hidden = Dense(128, activation='relu', trainable=True, name='output2_hidden0')(flattened2)
            output2_batchnorm = BatchNormalization(name='batch_normalization_output2')(output2_hidden)
            output2 = Dense(num_labels, activation='softmax', trainable=True, name='output2')(output2_batchnorm)

            outputs.append(output2)

        flattened3 = Flatten()(pooled12)
        output3_hidden = Dense(512, activation='relu', trainable=False, name='dense_1')(flattened3)
        output3_batchnorm = BatchNormalization(name='batch_normalization_14')(output3_hidden)
        output3 = Dense(num_labels, activation='softmax', trainable=False, name='dense_2')(output3_batchnorm)

        outputs.append(output3)

        return outputs

    def make(self, model_mode: ModelMode):
        dataset_name = self.dataset.dataset_name
        assert dataset_name in ('cifar10', 'cifar100'), 'Can only use the VGG model for cifar 10 and cifar 100'
        self._cifar_mode = CifarMode.CIFAR_10 if dataset_name == 'cifar10' else CifarMode.CIFAR_100

        super().make(model_mode=model_mode)

        pretrained_name = 'cifar10vgg' if self._cifar_mode == CifarMode.CIFAR_10 else 'cifar100vgg'
        dir_name = os.path.dirname(__file__)
        self._model.load_weights(os.path.join(dir_name, 'pretrained/{}.h5'.format(pretrained_name)), by_name=True)
