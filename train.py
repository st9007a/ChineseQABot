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

        logits = model(features['q'], features['a'])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=logits,
                export_outputs={
                    'response': tf.estimator.export.PredictOutput(logits)
                })

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(xentropy)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def train_input_fn():

    def _parse_example(serialized_example):
        data_fields = {'q': tf.FixedLenFeature((100,), tf.int64), 'a': tf.FixedLenFeature((100,), tf.int64)}
        parsed = tf.parse_single_example(serialized_example, data_fields)

        return {'q': parsed['q'], 'a': parsed['a']}, parsed['a']

    dataset = tf.data.TFRecordDataset('data/qa.tfrecords')
    dataset = dataset.map(_parse_example, num_parallel_calls=4)
    dataset = dataset.shuffle(50000)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)

    return dataset

def serving_input_fn():
    inputs = {'q': tf.placeholder(tf.int64, [None, 100]), 'a': tf.placeholder(tf.int64, [None, 100])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == '__main__':

    params, vocab = load_config()

    config = tf.estimator.RunConfig(save_checkpoints_steps=5000, model_dir='build/')
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params['arch'], config=config)

    for i in range(10):
        estimator.train(train_input_fn, steps=100000)
        estimator.export_savedmodel(export_dir_base='serve/', serving_input_receiver_fn=serving_input_fn)
