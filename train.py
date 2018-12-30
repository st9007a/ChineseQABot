#!/usr/bin/env python3
import sys
import yaml
from pprint import pprint

import tensorflow as tf

from model.transformer import Transformer

CONFIG = sys.argv[1]

def load_config():

    vocab = []

    with open(CONFIG, 'r') as f:
        params = yaml.load(f)

    with open('data/vocab.txt', 'r') as f:
        for line in f:
            vocab.append(line[:-1])

    params['arch']['vocab_size'] = len(vocab)

    return params, vocab

def model_fn(features, labels, mode, params):

    with tf.variable_scope('model'):
        model = Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

        logits = model(features, labels)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=logits,
                export_outputs={
                    'response': tf.estimator.export.PredictOutput(logits)
                })

        xentropy = tf.nn.softmax_cross_enctropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_sum(xentropy)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

if __name__ == '__main__':

    params, vocab = load_config()

    tf.estimator.Estimator(model_fn=model_fn, model_dir='build/', params=params['arch'])

    for i in range(10):
        #!TODO: train hook (batch size)
        estimator.train(dataset.train_input_fn, steps=100000)

