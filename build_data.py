#!/usr/bin/env python3
import sys
import string
import re

import tensorflow as tf
from zhon.hanzi import punctuation

PAD_ID = 0
PAD_TOKEN = '<p/>'
START_ID = 1
START_TOKEN = '<s>'
END_ID = 2
END_TOKEN = '</s>'

MAX_LENGTH = int(sys.argv[1])

def encode(line, char_dict):
    ret = []
    ret.append(START_ID)


    for i, w in enumerate(line):
        ret.append(char_dict[w])

    ret.append(END_ID)

    while len(ret) < MAX_LENGTH:
        ret.append(PAD_ID)

    return ret[:MAX_LENGTH]

def int64_tfrecord_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def find_and_validate(line):
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', line))

if __name__ == '__main__':

    vocab = {}

    with open('data/Gossiping-QA-Dataset.txt', 'r') as f:
        for line in f:
            line = find_and_validate(line)

            if len(line) == 0:
                continue

            for w in line:

                if w not in vocab:
                    vocab[w] = 0

                vocab[w] += 1

    vocab = [(k, vocab[k]) for k in vocab]
    vocab.sort(key=lambda x: x[1])
    vocab.reverse()
    vocab = [el[0] for el in vocab]
    char2idx = {w: i + 3 for i, w in enumerate(vocab)}

    print('vocab size (char level):', len(vocab))

    with open('data/vocab.txt', 'w') as f:
        f.write(PAD_TOKEN + '\n')
        f.write(START_TOKEN + '\n')
        f.write(END_TOKEN + '\n')

        for w in vocab:
            f.write(w + '\n')

    tf_writer = tf.python_io.TFRecordWriter('data/qa.tfrecords')

    with open('data/Gossiping-QA-Dataset.txt', 'r') as f:
        for line in f:
            line = line[:-1]
            q, a = line.split('\t')

            q = find_and_validate(q)
            a = find_and_validate(a)

            if len(q) == 0 or len(a) == 0:
                continue

            q = encode(q, char2idx)
            a = encode(a, char2idx)

            example = tf.train.Example(features=tf.train.Features(feature={
                'q': int64_tfrecord_feature(q),
                'a': int64_tfrecord_feature(a),
            }))

            tf_writer.write(example.SerializeToString())

    tf_writer.close()
