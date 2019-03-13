#!/usr/bin/env python3
from math import log, pow

import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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

        alive_seq.append(([], 0, res))

        for i in range(self.max_length):
            answer = np.array([el[2] for el in alive_seq])
            question = np.tile(input_arr, (answer.shape[0], 1))
            decode_ids = self._run([question, answer])

            new_alive_seq = []

            for state, decode_proba in zip(alive_seq, decode_ids[:, i, :]):
                candidate = np.argsort(decode_proba)[-self.beam_width:]
                decode_proba = softmax(decode_proba)

                for idx in candidate:
                    proba = log(decode_proba[idx])
                    seq = state[0] + [proba]
                    len_norm = pow((5.+i+1.) / 6, self.alpha)
                    score = sum(seq) / len_norm

                    new_res = np.array(state[2])
                    new_res[i] = idx

                    if idx == self.eos_id:
                        finished_seq.append((score, new_res))
                    else:
                        new_alive_seq.append((seq, score, new_res))

            alive_seq = new_alive_seq
            alive_seq.sort(key=lambda x: x[1], reverse=True)
            alive_seq = alive_seq[:self.beam_width]

            if len(finished_seq) > 0:
                finished_seq.sort(key=lambda x: x[0], reverse=True)
                finished_seq = finished_seq[:self.beam_width]

                if finished_seq[-1][0] > alive_seq[0][1]:
                    break

        return finished_seq
