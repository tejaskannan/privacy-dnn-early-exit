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


def dropout(x: tf2.Tensor, keep_rate: tf2.Tensor) -> tf2.Tensor:
    rand_mask = tf1.random.stateless_uniform(shape=tf2.shape(x), minval=0.0, maxval=1.0, seed=(381, 1031))
    rand_mask = tf2.stop_gradient(rand_mask)
    mask = tf2.cast(rand_mask < keep_rate, dtype=x.dtype)

    masked_x = x * mask
    return masked_x * (1.0 / keep_rate)
