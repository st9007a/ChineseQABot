#!/usr/bin/env python3
import tensorflow as tf

from .attention import Attention, SelfAttention
from .embedding import ShareWeightsEmbedding
from .ffn import FeedForwardNetwork
from .utils import get_padding, get_padding_bias, get_decoder_self_attention_bias, get_position_encoding

from .lib import beam_search

class Transformer():

    def __init__(self, params, train):
        self.params = params
        self.train = train

        self.embedding_softmax_layer = ShareWeightsEmbedding(
            params['vocab_size'], params['hidden_size'])

        self.encoder_stack = EncoderStack(params, train)
        self.decoder_stack = DecoderStack(params, train)

    def __call__(self, inputs, targets=None):
        initializer = tf.variance_scaling_initializer(self.params["initializer_gain"],
                                                      mode="fan_avg", distribution="uniform")

        with tf.variable_scope('transformer', initializer=initializer):
            attention_bias = get_padding_bias(inputs)

            encoder_outputs = self.encode(inputs, attention_bias)

            if target is None:
                return self.predict(encoder_output, attention_bias)

            else:
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

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.params["hidden_size"])
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
            ids: Current decoded sequences.
              int tensor with shape [batch_size * beam_size, i + 1]
            i: Loop index
            cache: dictionary of values storing the encoder output, encoder-decoder
              attention bias, and previous decoder attention values.
          Returns:
            Tuple of
              (logits with shape [batch_size * beam_size, vocab_size],
               updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache
        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_hidden_layers"])}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            alpha=self.params["alpha"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

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

        self.output_normlization = LayerNormalization(params['hidden_size'])

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

            enc_dec_attention_layer = SelfAttention(params['hidden_size'], params['num_heads'],
                                                    params['attention_dropout'], train)

            feed_forward_network = FeedForwardNetwork(params['hidden_size'], params['filter_size'],
                                                      params['relu_dropout'], train,
                                                      params['allow_ffn_pad'])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)
            ])

        self.output_normlization = LayerNormalization(params['hidden_size'])

    def __call__(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                 attention_bias, cache=None):

        for n, layer in enumerate(self.layers):
            self_attention = layer[0]
            enc_dec_attention = layer[1]
            feed_forward_network = layer[2]

            layer_name = 'layer_%d' % n
            layer_cache = cache[layer_name] if cache is not None else None

            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)

                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, encoder_outputs, attention_bias)

                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
