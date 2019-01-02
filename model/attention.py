#!/usr/bin/env python3
import tensorflow as tf

class Attention():

    def __init__(self, hidden_size, num_heads, attention_dropout, train):

        if hidden_size % num_heads != 0:
            raise ValueError('Hidden size must be evenly divisible by the number of heads.')

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="output_transform")

    def split_head(self, x):

        with tf.name_scope('split_head'):
            shape = tf.shape(x)
            batch_size = shape[0]
            length = shape[1]

            depth = (self.hidden_size // self.num_heads)

            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            return tf.transpose(x, [0, 2, 1, 3])

    def combine_head(self, x):

        with tf.name_scope('combine_head'):
            shape = tf.shape(x)
            batch_size = shape[0]
            length = shape[2]

            x = tf.transpose(x, [0, 2, 1, 3])

            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def __call__(self, x, y, bias):

        q = self.q_dense_layer(x)
        k = self.q_dense_layer(y)
        v = self.q_dense_layer(y)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = tf.matmul(q, v, transpose_b=True) + bias
        weights = tf.nn.softmax(logits, name='attention_weights')

        if self.train:
            weights = tf.nn.dropout(weights, 1 - self.attention_dropout)

        attention_output = tf.matmul(weights, v)
        attention_output = self.combine_head(attention_output)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output

class SelfAttention(Attention):

    def __call__(self, x, bias):
        return super(SelfAttention, self).__call__(x, x, bias)
