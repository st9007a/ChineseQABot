#!/usr/bin/env python3
from math import log, pow
import bisect

import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def insort(array, item, key=None):
    if len(array) == 0:
        array.append(item)
        return

    index_array = [key(el) for el in array] if key else array
    index_item = key(item) if key else item

    index = bisect.bisect_right(index_array, index_item)
    array.insert(index, item)

class BeamSearch():

    def __init__(self, session, eval_tensors, feed_tensors, alpha=0.6, beam_width=10, max_length=100, eos_id=1):
        self.session = session
        self.eval_tensors = eval_tensors
        self.feed_tensors = feed_tensors

        self.beam_width = beam_width
        self.max_length = max_length
        self.alpha = alpha
        self.eos_id = eos_id

    def _run(self, vals):
        feed_dict = {tensor: value for tensor, value in zip(self.feed_tensors, vals)}
        return self.session.run(self.eval_tensors, feed_dict=feed_dict)

    def search(self, input_arr):
        res = np.zeros((self.max_length,))
        alive_seq = []
        finished_seq = []

        alive_seq.append({'score': 0, 'probs': [], 'ids': res})

        for i in range(self.max_length):
            answer = np.array([el['ids'] for el in alive_seq])
            question = np.tile(input_arr, (answer.shape[0], 1))

            # shape of `decode_ids` = (batch_size, max_length, vocab_size)
            decode_ids = self._run([question, answer])

            new_alive_seq = []

            for state, decode_proba in zip(alive_seq, decode_ids[:, i, :]):

                # shape of `decode_proba` = (vocab_size,)
                candidate = np.argsort(decode_proba)[-self.beam_width:]
                decode_proba = softmax(decode_proba)

                for idx in candidate:
                    proba = log(decode_proba[idx])
                    seq = state['probs'] + [proba]

                    # See http://opennmt.net/OpenNMT/translation/beam_search/#length-normalization
                    len_norm = pow((5.+i+1.) / 6, self.alpha)
                    score = sum(seq) / len_norm

                    new_res = np.array(state['ids'])
                    new_res[i] = idx

                    if idx == self.eos_id:
                        finished_seq.append({'score': score, 'ids': new_res})
                    else:
                        new_alive_seq.append({'score': score, 'probs': seq, 'ids': new_res})

            alive_seq = new_alive_seq
            alive_seq.sort(key=lambda el: el['score'])
            alive_seq = alive_seq[-self.beam_width:]

            if len(finished_seq) > 0:
                finished_seq.sort(key=lambda el: el['score'])
                finished_seq = finished_seq[-self.beam_width:]

                if finished_seq[0]['score'] > alive_seq[-1]['score']:
                    break

        return finished_seq if len(finished_seq) > 0 else alive_seq

if __name__ == '__main__':

    arr = []

    insort(arr, (100, 'a'), key=lambda x: x[0])
    print(arr)
    insort(arr, (300, 'a'), key=lambda x: x[0])
    print(arr)
    insort(arr, (200, 'a'), key=lambda x: x[0])
    print(arr)
