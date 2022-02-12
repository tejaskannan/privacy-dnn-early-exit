import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from typing import Callable, Optional
from privddnn.utils.constants import SMALL_NUMBER


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
    elif name == 'tanh':
        return tf2.nn.tanh
    else:
        raise ValueError('No activation function with name: {}'.format(name))


def batch_normalization(inputs: tf2.Tensor, name: str) -> tf2.Tensor:
    num_channels = inputs.get_shape()[-1]

    with tf1.variable_scope(name):
        mean, variance = tf2.nn.moments(x=inputs, axes=[0, 1, 2], keepdims=True)

        offset = tf1.get_variable(name='offset',
                                  shape=(num_channels, ),
                                  initializer=tf2.random_normal_initializer(),
                                  dtype=inputs.dtype,
                                  trainable=True)

        scale = tf1.get_variable(name='scale',
                                 shape=(num_channels, ),
                                 initializer=tf2.random_normal_initializer(),
                                 dtype=inputs.dtype,
                                 trainable=True)

        return tf2.nn.batch_normalization(x=inputs,
                                          mean=mean,
                                          variance=variance,
                                          offset=offset,
                                          scale=scale,
                                          variance_epsilon=SMALL_NUMBER)


@tf2.custom_gradient
def differentiable_abs(x: tf2.Tensor) -> tf2.Tensor:

    def grad(dy: tf2.Tensor):
        factor = tf2.where(x < 0, -1.0, 1.0)
        is_zero = tf2.cast(tf2.less(tf2.abs(x), SMALL_NUMBER), dtype=factor.dtype)
        factor = (1.0 - is_zero) * factor

        return dy * factor

    return tf2.abs(x), grad


def dropout(x: tf2.Tensor, keep_rate: tf2.Tensor) -> tf2.Tensor:
    rand_mask = tf1.random.stateless_uniform(shape=tf2.shape(x), minval=0.0, maxval=1.0, seed=(381, 1031))
    rand_mask = tf2.stop_gradient(rand_mask)
    mask = tf2.cast(rand_mask < keep_rate, dtype=x.dtype)

    masked_x = x * mask
    return masked_x * (1.0 / keep_rate)
