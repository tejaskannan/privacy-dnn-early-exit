"""
Implements a dense feed-forward neural network using numpy. This model
is the top portion of a branchynet DNN.
"""
import numpy as np
from typing import Dict


def relu(x: np.ndarray):
    return np.maximum(x, 0)


class DenseNeuralNetwork:

    def __init__(self, model_weights: Dict[str, np.ndarray]):
        # Unpack the model parameters
        self._W1 = model_weights['hidden2/W:0'].T
        self._b1 = model_weights['hidden2/b:0'].reshape(-1, 1)

        self._W2 = model_weights['hidden3/W:0'].T
        self._b2 = model_weights['hidden3/b:0'].reshape(-1, 1)

        self._Wout = model_weights['output2/W:0'].T
        self._bout = model_weights['output2/b:0'].reshape(-1, 1)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        assert len(inputs.shape) == 2, 'Must provide a 2d input.'
        assert inputs.shape[1] == 1, 'Second dimension must be 1.'

        # Execute the hidden layers
        hidden1 = np.matmul(self._W1, inputs) + self._b1
        hidden1 = relu(hidden1)

        hidden2 = np.matmul(self._W2, hidden1) + self._b2
        hidden2 = relu(hidden2)

        # Execute the output layer
        logits = np.matmul(self._Wout, hidden2) + self._bout

        # Return the log probabilities
        return logits
