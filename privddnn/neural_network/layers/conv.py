import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from typing import Tuple

from .layer_utils import get_activation_fn


def fitnet_block(inputs: tf2.Tensor, num_filters: int, pool_size: int, pool_stride: int, name: str) -> Tuple[tf2.Tensor, tf2.Tensor, tf2.Tensor]:
    with tf1.variable_scope(name):
        # Create the convolution layers
        block_one = conv2d(inputs=inputs,
                           filter_size=3,
                           stride=1,
                           num_filters=num_filters,
                           activation='relu',
                           name='conv1')

        block_two = conv2d(inputs=block_one,
                           filter_size=3,
                           stride=1,
                           num_filters=num_filters,
                           activation='relu',
                           name='conv2')

        pooled = tf2.nn.max_pool(input=block_two,
                                 ksize=pool_size,
                                 strides=[1, pool_stride, pool_stride, 1],
                                 padding='SAME')

        return pooled, block_one, block_two



def weighted_add(x: tf2.Tensor, y: tf2.Tensor, name: str) -> tf2.Tensor:
    """
    Returns the sum w1 * x + w2 * y where w1 and w2 are trainable parameters.

    Args:
        x: A [B, H, W, C] tensor
        y: A [B, H, W, C] tensor
        name: The name of this layer
    Returns:
        A [B, H, W, C] tensor
    """
    with tf1.variable_scope(name):
        w1 = tf1.get_variable(shape=(),
                              dtype=x.dtype,
                              trainable=True,
                              initializer=tf1.glorot_uniform_initializer(),
                              name='weight-one')

        w2 = tf1.get_variable(shape=(),
                              dtype=x.dtype,
                              trainable=True,
                              initializer=tf1.glorot_uniform_initializer(),
                              name='weight-two')

        return w1 * x + w2 * y


def bottleneck_block(inputs: tf2.Tensor, filter_size: int, filter_stride: int, channel_mult: int, expand_size: int, squeeze_size: int, name: str) -> tf2.Tensor:
    """
    Creates a MobileNetv2 bottleneck block.

    Args:
        inputs: A [B, H, W, C] or [B, H, W] tensor of input features
        filter_size: The size to use for depthwise convolution filters
        filter_stride: The stride to use for depthwise convolutions
        channel_mult: The channel multiplier for depthwise convolutions
        expand_size: The size to use during expansion
        squeeze_size: The size to use during contraction
        name: The name of this layer
    Returns:
        The transformed tensor, [B, H, W, D]
    """
    with tf1.variable_scope(name):
        # Create the expansion layer
        expanded = conv2d(inputs, num_filters=expand_size, filter_size=1, stride=1, activation='relu6', name='conv-expansion')

        # Apply the depthwise convolution
        depthwise = depthwise_conv2d(expanded, filter_size=filter_size, channel_mult=channel_mult, stride=filter_stride, activation='relu6', name='depthwise')

        # Apply the contraction layer
        contracted = conv2d(depthwise, num_filters=squeeze_size, filter_size=1, stride=1, activation='relu6', name='contracted')

        # Create the residual connection (TODO: Handle dimension changes)
        residual = conv2d(inputs, num_filters=squeeze_size, filter_size=1, stride=1, activation='linear', name='residual')

        return tf2.add(contracted, residual)


def conv2d(inputs: tf2.Tensor,
           num_filters: int,
           filter_size: int,
           stride: int,
           activation: str,
           name: str) -> tf2.Tensor:
    """
    Creates 2d convolutional neural network layer

    Args:
        inputs: A [B, H, W] or [B, H, W, C] tensor of inputs features
        filter_size: The height and width of each filter
        num_filters: The number of output filters (K)
        stride: The stride to use when convolving
        activation: The name of the activaton function
        name: The name of this layer
    """
    input_shape = inputs.get_shape()
    ndims = len(input_shape)

    assert ndims in (3, 4), 'Must provide an input with 3 or 4 dimensions. Got: {}'.format(ndims)

    # Add a single channel for 3d inputs
    if ndims == 3:
        inputs = tf2.expand_dims(inputs, axis=-1)  # [B, H, W, 1]

    input_channels = inputs.get_shape()[-1]

    with tf1.variable_scope(name):
        # Create the trainable variables
        filters = tf1.get_variable(shape=[filter_size, filter_size, input_channels, num_filters],
                                   dtype=inputs.dtype,
                                   initializer=tf1.glorot_uniform_initializer(),
                                   name='filters')

        bias = tf1.get_variable(shape=[1, 1, 1, num_filters],
                                dtype=inputs.dtype,
                                initializer=tf1.glorot_uniform_initializer(),
                                name='bias')

        # Apply the convolution filters
        conv_transformed = tf2.nn.conv2d(input=inputs,
                                         filters=filters,
                                         strides=stride,
                                         padding='SAME')

        conv_transformed = tf2.add(conv_transformed, bias)

        # Apply the activation function if needed
        activation_fn = get_activation_fn(activation)
        transformed = activation_fn(conv_transformed) if activation_fn is not None else conv_transformed

        return transformed


def depthwise_conv2d(inputs: tf2.Tensor,
                     filter_size: int,
                     channel_mult: int,
                     stride: int,
                     activation: str,
                     name: str) -> tf2.Tensor:
    """
    Creates a depthwise convolution layer.

    Args:
        inputs: A [B, H, W] or [B, H, W, C] tensor of input features
        filter_size: The height and width of each convolution filter
        channel_mult: The number of output channels for each filter (D)
        stride: The filter stride
        activation: The name of the activation function
        name: The name of this layer
    Returns:
        A transformed tensor of size [B, H, W, D * C]
    """
    input_shape = inputs.get_shape()
    ndims = len(input_shape)

    assert ndims in (3, 4), 'Must provide an input with 3 or 4 dimensions. Got: {}'.format(ndims)

    # Add a single channel for 3d inputs
    if ndims == 3:
        inputs = tf2.expand_dims(inputs, axis=-1)  # [B, H, W, 1]

    input_channels = inputs.get_shape()[-1]

    with tf1.variable_scope(name):
        # Create the trainable variables
        filters = tf1.get_variable(shape=[filter_size, filter_size, input_channels, channel_mult],
                                   dtype=inputs.dtype,
                                   initializer=tf1.glorot_uniform_initializer(),
                                   name='filters')

        bias = tf1.get_variable(shape=[1, 1, 1, input_channels * channel_mult],
                                dtype=inputs.dtype,
                                initializer=tf1.glorot_uniform_initializer(),
                                name='bias')

        # Apply the depthwise convolution, [B, H, W, D * C]
        conv_transformed = tf2.nn.depthwise_conv2d(input=inputs,
                                                   filter=filters,
                                                   strides=[1, stride, stride, 1],
                                                   padding='SAME')

        # Add the bias
        conv_transformed = tf2.add(conv_transformed, bias)

        # Apply the activation function
        activation_fn = get_activation_fn(activation)
        transformed = activation_fn(conv_transformed) if activation_fn is not None else conv_transformed

        return transformed
