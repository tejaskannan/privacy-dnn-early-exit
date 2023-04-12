"""
Implements a dense neural network in NumPy outside of Keras
for execution of only the latter part of the split DNN.
"""
import numpy as np
import h5py
from collections import namedtuple
from argparse import ArgumentParser


Prediction = namedtuple('Prediction', ['pred', 'logits'])


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


class DenseNeuralNetwork:

    def __init__(self, weights_path: str):
        self._layer_names: List[str] = ['dense_1', 'dense_2', 'dense_3', 'output1']
        self._weight_matrices: Dict[str, np.ndarray] = dict()
        self._weight_biases: Dict[str, np.ndarray] = dict()

        with h5py.File(weights_path, 'r') as fin:
            model_weights = fin['model_weights']

            for layer_name in self._layer_names:
                layer = model_weights[layer_name][layer_name]
                self._weight_matrices[layer_name] = layer['kernel:0'][:].T
                self._weight_biases[layer_name] = layer['bias:0'][:]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        assert len(inputs.shape) == 1, 'Must provide a 1d array of inputs'
        state = inputs

        for layer_name in self._layer_names:
            state = np.matmul(self._weight_matrices[layer_name], state) + self._weight_biases[layer_name]

            if 'output' not in layer_name:
                state = relu(state)

        return Prediction(logits=state, pred=int(np.argmax(state)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights-path', type=str, required=True, help='Path to the saved DNN parameters. Only supports dense branchynet models.')
    args = parser.parse_args()

    dnn = DenseNeuralNetwork(args.weights_path)

    inputs = np.random.uniform(size=(316, ))
    prediction = dnn(inputs)
    print(prediction)
