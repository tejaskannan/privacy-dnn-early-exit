import tensorflow.compat.v1 as tf


def dense_relu(self, inputs: tf.Tensor, units: int, name: str):
    input_units = inputs.get_shape()[-1]

    with tf.variable_scope(name):
        weight_mat = tf.get_variable(name='kernel:0',
                                     shape=(input_units, units),
                                     initializer=tf.glorot_uniform_initializer(),
                                     trainable=False)

        bias = tf.get_variable(name='bias:0',
                               shape=(units, ),
                               initializer=tf.glorot_uniform_initializer(),
                               trainable=False)

        linear_transformed = tf.matmul(inputs, weight_mat) + bias
        return tf.nn.relu(linear_transformed)


