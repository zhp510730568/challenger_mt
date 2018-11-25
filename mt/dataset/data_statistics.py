import os

from collections import Counter

import tensorflow as tf

import nltk

from collections import Counter

root_path = '../data1'
en_corpus = 'test.txt'


def token_statistics():
    counter = Counter()
    data_path = os.path.join(root_path, en_corpus)
    with tf.gfile.Open(data_path) as f:
        for line in f:
            tokens = [token for token in nltk.word_tokenize(line.strip())]
            for token in tokens:
                counter[token] += 1
    for key, _ in counter.items():
        counter[key] = 0
    with tf.gfile.Open(os.path.join(root_path, 'en_corpus')) as f:
        for sentence in f:
            for token in sentence.split():
                if token in counter:
                    counter[token] += 1
    sorted_counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    for key,value in sorted_counter:
        print(key, value)


if __name__=='__main__':
    counter = Counter()
    with tf.gfile.Open(os.path.join(root_path, 'train.txt')) as f:
        for line in f:
            arr = line.strip().split('\t')
            sentence = arr[2]
            for index in range(len(sentence) - 3):
                subword = sentence[index: index + 3]
                counter[subword] += 1
        sorted_counter = sorted(counter.items(), key=lambda item: item[1], reverse=False)
        for key,value in sorted_counter:
            print(key, value)