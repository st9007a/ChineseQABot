#!/usr/bin/env python3
import sys
import time

import numpy as np
import tensorflow as tf

from utils.beam_search import BeamSearch

def load_model(model_dir):
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
    signature = meta_graph_def.signature_def

    q_name = signature['serving_default'].inputs['q'].name
    a_name = signature['serving_default'].inputs['a'].name

    output_name = signature['serving_default'].outputs['output'].name

    q = sess.graph.get_tensor_by_name(q_name)
    a = sess.graph.get_tensor_by_name(a_name)
    output = sess.graph.get_tensor_by_name(output_name)

    return sess, {'q': q, 'a': a, 'output': output}

def load_vocab():
    word_list = []
    word_idx_map = {}

    with open('data/vocab.txt', 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            word_idx_map[line] = len(word_list)
            word_list.append(line)

    return word_list, word_idx_map

if __name__ == '__main__':

    idx2word, word2idx = load_vocab()
    sess, tensors = load_model(sys.argv[1])
    beam_search = BeamSearch(session=sess, eval_tensors=tensors['output'], feed_tensors=[tensors['q'], tensors['a']])

    while True:
        test_input = input('請輸入中文句子:')

        test_input = [word2idx[el] for el in test_input if el in word2idx]
        while len(test_input) < 100:
            test_input.append(0)

        start = time.time()
        _, finished_seq = beam_search.search(test_input)
        end = time.time()

        print('Search time: %.6f sec' % (end - start))

        finished_seq.sort(key=lambda x: x[0], reverse=True)

        for seq in finished_seq:
            response = [idx2word[int(el)] for el in seq[1]]
            while response[-1] == '<pad>' or response[-1] == '<eos>':
                response.pop()

            response = ''.join(response)
            print('Score: % .6f, Response: %s' % (seq[0], response))

        print('==============')