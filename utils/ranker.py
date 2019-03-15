#!/usr/bin/env python3
from math import pow, log

import numpy as np

stopwords = ['嗎', '啊', '啦', '喇', '不', '是', '好', '吧', '你', '就', '的']

class Ranker():

    def __init__(self, repeat_penality=1.5):
        self.repeat_penality = repeat_penality

    def fit_transform(self, sequences):
        tf = []
        df = {}
        token_set = set()
        avg_len = 0.
        scores = []

        for seq in sequences:
            ttf = {}
            exist = set()
            for token in seq:
                if token in stopwords:
                    continue

                ttf[token] = ttf.get(token, 0) + 1
                token_set.add(token)

                if token not in exist:
                    df[token] = df.get(token, 0) + 1
                    exist.add(token)

            tf.append(ttf)
            avg_len += len(seq)

        avg_len /= len(sequences)

        for i, seq in enumerate(sequences):
            score = 0
            tokens = set()
            for token in seq:
                if token in stopwords:
                    continue

                s = df[token] * 1.0 / pow(tf[i][token], self.repeat_penality)
                tokens.add(token)

                score += s

            score += len(tokens)/len(token_set)
            scores.append(score)

        return scores

if __name__ == '__main__':
    corpus = [
        '麥克風測試',
        '測試',
    ]

    ranker = Ranker()
    s = ranker.fit_transform(corpus)
    print(s)
