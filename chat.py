#!/usr/bin/env python3
import sys

import numpy as np
import tensorflow as tf

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

    test_input = '有沒有主管的八掛'

    test_input = [word2idx[el] for el in test_input if el in word2idx]
    while len(test_input) < 100:
        test_input.append(0)
    a = np.zeros((1, 100))

    for i in range(100):
        predict = sess.run(tensors['output'], feed_dict={tensors['q']: np.array([test_input]), tensors['a']: a})
        a[0, i] = np.argmax(predict[0, i])

        if a[0, i] == 1:
            break

    response = [idx2word[int(el)] for el in a[0]]
    while response[-1] == '<pad>' or response[-1] == '<eos>':
        response.pop()

    print(''.join(response))
