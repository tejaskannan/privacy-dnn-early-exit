import h5py
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.layers import BatchNormalization, Concatenate
from typing import List, Tuple, Dict, Any


class JetsonNanoBranchyNetDNN:

    def __init__(self, model_path: str, input_shape: Tuple[int, ...], num_labels: int):
        self._num_labels = num_labels

        self._model0, self._model1 = self.make(input_shape=input_shape,
                                               num_labels=num_labels)

        with h5py.File(model_path, 'r') as fin:
            model_weights = fin['model_weights']

            model0_layers = ['dense', 'output0']
            for layer_name in model0_layers:
                kernel = model_weights[layer_name][layer_name]['kernel:0']
                bias = model_weights[layer_name][layer_name]['bias:0']
                self._model0.get_layer(layer_name).set_weights([kernel, bias])

            model1_layers = ['dense_1', 'dense_2', 'dense_3', 'output1']
            for layer_name in model1_layers:
                kernel = model_weights[layer_name][layer_name]['kernel:0']
                bias = model_weights[layer_name][layer_name]['bias:0']
                self._model1.get_layer(layer_name).set_weights([kernel, bias])

    @property
    def name(self) -> str:
        return 'branchynet-dnn'

    @property
    def num_outputs(self) -> int:
        return 2

    def make(self, input_shape: Tuple[int, ...], num_labels: int) -> Tuple[Model, Model]:
        # Make the early exit model
        model0_inputs = Input(shape=input_shape, name='model0-inputs')

        hidden0 = Dense(16, activation='relu', name='dense')(model0_inputs)
        output0 = Dense(num_labels, activation='softmax', name='output0')(hidden0)

        model0 = Model(inputs=model0_inputs, outputs=[hidden0, output0])
        model0.compile(loss=['mse', 'mse'])

        # Make the full model
        model1_inputs0 = Input(shape=input_shape, name='model1-inputs0')
        model1_inputs1 = Input(shape=hidden0.get_shape()[1:], name='model1-inputs1')
        concat = Concatenate(axis=-1)([model1_inputs0, model1_inputs1])

        hidden1 = Dense(128, activation='relu', name='dense_1')(concat)
        hidden2 = Dense(128, activation='relu', name='dense_2')(hidden1)
        hidden3 = Dense(128, activation='relu', name='dense_3')(hidden2)
        output1 = Dense(num_labels, activation='softmax', name='output1')(hidden3)

        model1 = Model(inputs=[model1_inputs0, model1_inputs1], outputs=[hidden3, output1])
        model1.compile(loss=['mse', 'mse'])

        return model0, model1

    def execute_early_exit(self, inputs: np.ndarray) -> Tuple[np.ndarray]:
        return self._model0.predict(inputs)

    def execute_full_model(self, inputs: np.ndarray, model0_state: np.ndarray) -> Tuple[np.ndarray]:
        return self._model1.predict([inputs, model0_state])
