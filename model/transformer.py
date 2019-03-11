#!/usr/bin/env python3
import tensorflow as tf

from .attention import Attention, SelfAttention
from .embedding import ShareWeightsEmbedding
from .ffn import FeedForwardNetwork
from .utils import get_padding, get_padding_bias, get_decoder_self_attention_bias, get_position_encoding

class Transformer():

    def __init__(self, params, train):
        self.params = params
        self.train = train

        self.embedding_softmax_layer = ShareWeightsEmbedding(
            params['vocab_size'], params['hidden_size'])

        self.encoder_stack = EncoderStack(params, train)
        self.decoder_stack = DecoderStack(params, train)

    def __call__(self, inputs, targets):
        initializer = tf.variance_scaling_initializer(self.params["initializer_gain"],
                                                      mode="fan_avg", distribution="uniform")

        with tf.variable_scope('transformer', initializer=initializer):
            attention_bias = get_padding_bias(inputs)

            with tf.variable_scope('encode_stack'):
                encoder_outputs = self.encode(inputs, attention_bias)

            with tf.variable_scope('decode_stack'):
                logits = self.decode(targets, encoder_outputs, attention_bias)
            return logits

    def encode(self, inputs, attention_bias):

        with tf.name_scope('encode'):

            embedded_inputs = self.embedding_softmax_layer(inputs)
            inputs_padding = get_padding(inputs)

            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = get_position_encoding(length, self.params['hidden_size'])

                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(encoder_inputs, 1. - self.params['layer_postprocess_dropout'])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):

        with tf.name_scope('decode'):
            decoder_inputs = self.embedding_softmax_layer(targets)

            with tf.name_scope('shift_targets'):
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += get_position_encoding(length, self.params['hidden_size'])

            if self.train:
                decoder_inputs = tf.nn.dropout(decoder_inputs, 1. - self.params['layer_postprocess_dropout'])

            decoder_self_attention_bias = get_decoder_self_attention_bias(length)

            outputs = self.decoder_stack(decoder_inputs, encoder_outputs,
                                         decoder_self_attention_bias, attention_bias)

            logits = self.embedding_softmax_layer.linear(outputs)

            return logits

class LayerNormalization():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, x, epsilon=1e-6):

        self.scale = tf.get_variable('layer_norm_scale', [self.hidden_size],
                                     initializer=tf.ones_initializer())

        self.bias = tf.get_variable('layer_norm_bias', [self.hidden_size],
                                    initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias

class PrePostProcessingWrapper():

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params['layer_postprocess_dropout']
        self.train = train

        self.layer_norm = LayerNormalization(params['hidden_size'])

    def __call__(self, x, *args, **kwargs):

        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)

        if self.train:
            y = tf.nn.dropout(y, 1. - self.postprocess_dropout)

        return x + y

class EncoderStack():

    def __init__(self, params, train):

        self.layers = []

        for _ in range(params['num_hidden_layers']):
            self_attention_layer = SelfAttention(params['hidden_size'], params['num_heads'],
                                                 params['attention_dropout'], train)

            feed_forward_network = FeedForwardNetwork(params['hidden_size'], params['filter_size'],
                                                      params['relu_dropout'], train,
                                                      params['allow_ffn_pad'])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)
            ])

        self.output_normalization = LayerNormalization(params['hidden_size'])

    def __call__(self, encoder_inputs, attention_bias, inputs_padding):

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope('layer_%d' % n):
                with tf.variable_scope('self_attention'):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)

                with tf.variable_scope('ffn'):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)

class DecoderStack():

    def __init__(self, params, train):

        self.layers = []

        for _ in range(params['num_hidden_layers']):
            self_attention_layer = SelfAttention(params['hidden_size'], params['num_heads'],
                                                 params['attention_dropout'], train)

            enc_dec_attention_layer = Attention(params['hidden_size'], params['num_heads'],
                                                params['attention_dropout'], train)

            feed_forward_network = FeedForwardNetwork(params['hidden_size'], params['filter_size'],
                                                      params['relu_dropout'], train,
                                                      params['allow_ffn_pad'])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)
            ])

        self.output_normalization = LayerNormalization(params['hidden_size'])

    def __call__(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias):

        for n, layer in enumerate(self.layers):
            self_attention = layer[0]
            enc_dec_attention = layer[1]
            feed_forward_network = layer[2]

            layer_name = 'layer_%d' % n

            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention(
                        decoder_inputs, decoder_self_attention_bias)

                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention(
                        decoder_inputs, encoder_outputs, attention_bias)

                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
