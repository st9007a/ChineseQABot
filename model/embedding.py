#!/usr/bin/env python3
import tensorflow as tf

class ShareWeightsEmbedding():

    def __init__(self, vocab_size, hidden_size):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def __call__(self, x):

        with tf.variable_scope('embedding_and_softmax', reuse=tf.AUTO_REUSE):
            self.share_weights = tf.get_variable(
                'weights',
                [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(0, self.hidden_size ** -0.5),
            )

        with tf.name_scope('embedding'):
            mask = tf.to_float(tf.not_equal(x, 0))

            embedding = tf.gather(self.share_weights, x)
            embedding *= tf.expand_dims(mask, -1)
            embedding *= self.hidden_size ** 0.5

            return embedding

    def linear(self, x):

        with tf.name_scope('presoftmax_linear'):
            shape = tf.shape(x)
            batch_size = shape[0]
            length = shape[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.share_weights, transpose_b=True)
            logits = tf.reshape(logits, [batch_size, length, self.vocab_size])

            return logits
