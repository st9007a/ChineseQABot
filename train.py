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

if __name__ == '__main__':

    params, vocab = load_config()
    pprint(params)

    model = Transformer(params['arch'], True)
