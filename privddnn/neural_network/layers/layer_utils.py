import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from typing import Callable, Optional


def get_activation_fn(name: str) -> Optional[Callable[[tf2.Tensor], tf2.Tensor]]:
    """
    Gets the activation function by name.
    """
    name = name.lower()

    if name == 'linear':
        return None
    elif name == 'relu':
        return tf1.nn.relu
    elif name == 'relu6':
        return tf1.nn.relu6
    elif name == 'leaky_relu':
        return tf1.nn.leaky_relu
    else:
        raise ValueError('No activation function with name: {}'.format(name))


@tf2.custom_gradient
def differentiable_abs(x: tf2.Tensor) -> tf2.Tensor:

    def grad(dy: tf2.Tensor):
        factor = tf2.where(x < 0, -1.0, 1.0)
        return dy * factor

    return tf2.abs(x), grad
