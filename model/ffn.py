#!/usr/bin/env python3
import tensorflow as tf

class FeedForwardNetwork():

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):

        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name='filter_layer',
        )

        self.output_dense_layer = tf.layers.Dense(
            hidden_size,
            use_bias=True,
            activation=tf.nn.relu,
            name='output_layer',
        )

    def __call__(self, x, padding=None):

        padding = None if not self.allow_pad else padding

        shape = tf.shape(x)
        batch_size = shape[0]
        length = shape[1]

        if padding is not None:
            with tf.name_scope('remove_padding'):

                pad_mask = tf.reshape(padding, [-1])
                nonpad_ids = tf.to_int(tf.where(pad_mask < 1e-9))

                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)

        if self.train:
            output = tf.nn.dropout(output, 1. - self.relu_dropout)

        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])

        return output
