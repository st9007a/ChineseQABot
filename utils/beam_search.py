#!/usr/bin/env python3
from math import log

import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class BeamSearch():

    def __init__(self, session, eval_tensors, feed_tensors, beam_width=10, max_length=100):
        self.session = session
        self.eval_tensors = eval_tensors
        self.feed_tensors = feed_tensors

        self.beam_width = beam_width
        self.max_length = max_length

    def _run(self, vals):
        feed_dict = {tensor: value for tensor, value in zip(self.feed_tensors, vals)}
        return self.session.run(self.eval_tensors, feed_dict=feed_dict)

    def search(self, input_arr):
        dead = 0
        res = np.zeros((self.max_length,))
        min_score = -10000
        states = []
        finished_seq = []

        states.append(([], 0, res))

        for i in range(self.max_length):
            answer = np.array([el[2] for el in states])
            question = np.tile(input_arr, (answer.shape[0], 1))
            decode_ids = self._run([question, answer])

            new_states = []

            for state, decode_proba in zip(states, decode_ids[:, i, :]):
                candidate = np.argsort(decode_proba)[-self.beam_width:]
                decode_proba = softmax(decode_proba)

                for idx in candidate:
                    proba = log(decode_proba[idx])
                    score = state[1] * len(state[0]) + proba
                    seq = state[0] + [proba]
                    score /= (i + 1)

                    new_res = np.array(state[2])
                    new_res[i] = idx

                    if idx == 1:
                        finished_seq.append((score, new_res))
                        dead += 1
                    else:
                        new_states.append((seq, score, new_res))

            if dead == self.beam_width:
                break

            states = new_states
            states.sort(key=lambda x: x[1], reverse=True)
            states = states[:self.beam_width-dead]
            min_score = states[-1][1]

            finished_seq.sort(key=lambda x: x[0], reverse=True)

            while len(finished_seq) > 0 and finished_seq[-1][0] < min_score:
                finished_seq.pop()
                dead -= 1

        return states, finished_seq
