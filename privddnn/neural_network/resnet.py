import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Layer, Add, AveragePooling2D
from tensorflow.keras.metrics import Metric
from typing import List, Dict, Any

from privddnn.classifier import ModelMode
from .constants import DROPOUT_KEEP_RATE, BATCH_SIZE, MetaName
from .early_exit_dnn import EarlyExitNeuralNetwork


def residual_block(block_input: Layer, filter_size: int, num_filters: int, dropout_rate: float, name: str):
    conv0 = Conv2D(filters=num_filters,
                   kernel_size=filter_size,
                   strides=(1, 1),
                   padding='same',
                   activation='linear',
                   name='{}/conv0'.format(name))(block_input)
    batchnorm0 = BatchNormalization()(conv0)
    conv0 = ReLU()(batchnorm0)
    dropout0 = Dropout(rate=dropout_rate)(conv0)

    conv1 = Conv2D(filters=num_filters,
                   kernel_size=filter_size,
                   strides=(1, 1),
                   padding='same',
                   activation='linear',
                   name='{}/conv1'.format(name))(dropout0)
    batchnorm1 = BatchNormalization()(conv1)
    conv1 = ReLU()(batchnorm1)

    residual_add = Add()([conv1, block_input])
    return residual_add


def skip_block(block_input: Layer, filter_size: int, num_filters: int, dropout_rate: float, name: str):
    conv0 = Conv2D(filters=num_filters,
                   kernel_size=filter_size,
                   strides=(2, 2),
                   padding='same',
                   activation='linear',
                   name='{}/conv0'.format(name))(block_input)
    batchnorm0 = BatchNormalization()(conv0)
    conv0 = ReLU()(batchnorm0)
    dropout0 = Dropout(rate=dropout_rate)(conv0)

    conv1 = Conv2D(filters=num_filters,
                   kernel_size=filter_size,
                   strides=(1, 1),
                   padding='same',
                   activation='linear',
                   name='{}/conv1'.format(name))(dropout0)
    batchnorm1 = BatchNormalization()(conv1)
    conv1 = ReLU()(batchnorm1)

    skip_conv = Conv2D(filters=num_filters,
                       kernel_size=filter_size,
                       strides=(2, 2),
                       padding='same',
                       activation='linear',
                       name='{}/skip_conv'.format(name))(block_input)
  
    result = Add()([skip_conv, conv1])
    return result


class ResNet18(EarlyExitNeuralNetwork):

    @property
    def name(self) -> str:
        return 'resnet18'

    @property
    def default_hypers(self) -> Dict[str, Any]:
        default_hypers = super().default_hypers
        default_hypers['num_outputs'] = 2
        return default_hypers

    @property
    def num_outputs(self) -> int:
        return self.hypers['num_outputs']

    def make_loss(self) -> Dict[str, str]:
        return {
            'output0': 'sparse_categorical_crossentropy',
            'output1': 'sparse_categorical_crossentropy'
        }

    def make_metrics(self) -> Dict[str, Metric]:
        return {
            'output0': tf.metrics.SparseCategoricalAccuracy('acc'),
            'output1': tf.metrics.SparseCategoricalAccuracy('acc')
        }

    def make_loss_weights(self) -> Dict[str, float]:
        return {
            'output0': 0.1,
            'output1': 0.9
        }

    def compute_probs(self, inputs: np.ndarray, should_approx: bool) -> np.ndarray:
        """
        Computes the predicted probabilites on the given dataset.
        """
        preds = self._model.predict(inputs, batch_size=self.hypers[BATCH_SIZE], verbose=0)
        return np.concatenate([np.expand_dims(arr, axis=1) for arr in preds], axis=1)

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> List[Layer]:
        dropout_rate = 0.0 if model_mode == ModelMode.TEST else (1.0 - self.hypers[DROPOUT_KEEP_RATE])

        conv0 = Conv2D(filters=64,
                       kernel_size=7,
                       strides=(2, 2),
                       padding='same',
                       activation='linear',
                       name='conv0')(inputs)
        conv0 = ReLU()(conv0)
        pooled0 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(conv0)

        block0 = residual_block(block_input=conv0, num_filters=64, filter_size=3, dropout_rate=dropout_rate, name='block0')
        block1 = residual_block(block_input=block0, num_filters=64, filter_size=3, dropout_rate=dropout_rate, name='block1')

        block2 = skip_block(block_input=block1, num_filters=128, filter_size=3, dropout_rate=dropout_rate, name='block2')
        block3 = residual_block(block_input=block2, num_filters=128, filter_size=3, dropout_rate=dropout_rate, name='block3')

        block4 = skip_block(block_input=block3, num_filters=256, filter_size=3, dropout_rate=dropout_rate, name='block4')
        block5 = residual_block(block_input=block4, num_filters=256, filter_size=3, dropout_rate=dropout_rate, name='block5')

        block6 = skip_block(block_input=block5, num_filters=512, filter_size=3, dropout_rate=dropout_rate, name='block6')
        block7 = residual_block(block_input=block6, num_filters=512, filter_size=3, dropout_rate=dropout_rate, name='block7')

        pooled0 = AveragePooling2D(pool_size=3, strides=2)(block1)
        flattened0 = Flatten()(pooled0)
        output0_hidden = Dense(units=256, activation='relu', name='output0_hidden')(flattened0)
        output0_dropout = Dropout(rate=0.2)(output0_hidden)
        output0 = Dense(units=10, activation='softmax', name='output0')(output0_dropout)

        pooled1 = GlobalAveragePooling2D()(block7)
        output1_hidden = Dense(units=1024, activation='relu', name='output1_hidden')(pooled1)
        output1_dropout = Dropout(rate=0.2)(output1_hidden)
        output1 = Dense(units=10, activation='softmax', name='output1')(output1_dropout)

        return [output0, output1]

    @classmethod
    def restore(cls, path: str, model_mode: ModelMode):
        """
        Restores the model saved at the given path
        """
        # Load the saved data
        metadata = {
            MetaName.DATASET_NAME: 'cifar10_corrupted',
            MetaName.INPUT_SHAPE: (32, 32, 3),
            MetaName.NUM_LABELS: 10
        }

        # Build the model
        model = cls(hypers=dict(), dataset_name=metadata[MetaName.DATASET_NAME])
        model._metadata = metadata
        model._is_metadata_loaded = True

        model.dataset.fit_normalizer()
        model.dataset.normalize_data()

        model.make(model_mode=model_mode)
        model._model.load_weights(path, by_name=True)

        return model
