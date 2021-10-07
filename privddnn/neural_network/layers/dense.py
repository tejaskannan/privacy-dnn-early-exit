import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from typing import Union

from .layer_utils import get_activation_fn, dropout


def dense(inputs: Union[tf2.Tensor, tf1.placeholder],
          output_units: int,
          dropout_keep_rate: tf1.placeholder,
          use_dropout: bool,
          activation: str,
          trainable: bool,
          name: str) -> tf2.Tensor:
    """
    Creates a dense (feed-forward) neural network layer

    Args:
        inputs: A [B, N] tensor with the input values
        output_units: The dimension of each output vector (M)
        dropout_keep_rate: The dropout rate to use after processing
        use_dropout: Whether to use dropout or not
        activation: The name of the activation function (e.g. relu)
        name: The name of this layer
    Returns:
        A [B, M] tensor containing the transformed values
    """
    input_units = inputs.get_shape()[-1]

    with tf1.variable_scope(name):
        # Create the trainable variables
        W = tf1.get_variable(shape=[input_units, output_units],
                             dtype=inputs.dtype,
                             name='W',
                             initializer=tf1.glorot_uniform_initializer(),
                             trainable=trainable)

        b = tf1.get_variable(shape=[1, output_units],
                             dtype=inputs.dtype,
                             name='b',
                             initializer=tf1.glorot_uniform_initializer(),
                             trainable=trainable)

        # Apply the linear transformation
        linear_transformed = tf2.add(tf2.matmul(inputs, W), b)

        # Apply the activation function
        activation_fn = get_activation_fn(activation)
        transformed = activation_fn(linear_transformed) if activation_fn is not None else linear_transformed

        # Apply drppout if needed
        if use_dropout:
            return dropout(transformed, keep_rate=dropout_keep_rate)

        return transformed
