import os
from collections import Counter

import nltk

import tensorflow as tf

root_path='../data1'
file_name='train.txt'

en_corpus_name='en_corpus'
ch_corpus_name='ch_corpus'
en_vocab_name='en_vocab'
ch_vocab_name='ch_vocab'


class T2TDataset():
    def __init__(self, data_path, en_corpus_path, ch_corpus_path, en_vocab_path, ch_vocab_path,
                 en_limit=100000, ch_limit=10000):
        if data_path is None:
            raise ValueError('data_path must not be None')
        if not os.path.exists(data_path):
            raise ValueError("file doesn't exists")

        self._data_path=data_path
        self._en_corpus_path = en_corpus_path
        self._ch_corpus_path = ch_corpus_path
        self._en_vocab_path = en_vocab_path
        self._ch_vocab_path = ch_vocab_path
        self._en_limit = en_limit
        self._ch_limit = ch_limit

        if (not os.path.exists(self._en_corpus_path)) or (not os.path.exists(self._ch_corpus_path)):
            self._export_corpus()

        if (not os.path.exists(self._en_vocab_path)) or (not os.path.exists(self._ch_vocab_path)):
            self._export_vocab()

    def _export_corpus(self):
        count = 0
        print('extract corpus start')
        with tf.gfile.Open(self._data_path, 'r') as f:
            with tf.gfile.Open(self._en_corpus_path, 'w') as en_file:
                with tf.gfile.Open(self._ch_corpus_path, 'w') as ch_file:
                    for sentence in f:
                        arr = sentence.strip().split('\t')
                        en_sentence = arr[2]
                        ch_sentence = arr[3]
                        en_tokens = nltk.word_tokenize(en_sentence)
                        ch_tokens = [token for token in ch_sentence]
                        en_file.write('%s\n' % (' '.join(en_tokens)))
                        ch_file.write('%s\n' % (' '.join(ch_tokens)))
                        count += 1
                        if count % 100000 == 0:
                            print('current process count: %d' % (count))
        print('extract corpus is over')

    def _export_vocab(self):
        if self._en_corpus_path is None or (not os.path.exists(self._en_corpus_path)):
            raise ValueError('en_corpus_path must exists')
        if self._ch_corpus_path is None or (not os.path.exists(self._ch_corpus_path)):
            raise ValueError('ch_corpus_path must exists')

        en_tokens_counter = Counter()
        with tf.gfile.Open(self._en_corpus_path, 'r') as en_file:
            for sentence in en_file:
                for token in sentence.split():
                    en_tokens_counter[token] += 1
        print(en_tokens_counter)
        sorted_en_list = sorted(en_tokens_counter.items(), key=lambda kv: kv[1], reverse=True)
        with open(self._en_vocab_path, 'w') as f:
            for k, v in sorted_en_list[0: self._en_limit]:
                f.write('%s\n' % (k))

        print('english tokens count is %d' % (len(en_tokens_counter)))
        ch_tokens_counter = Counter()
        with tf.gfile.Open(self._ch_corpus_path, 'r') as ch_file:
            for sentence in ch_file:
                for token in sentence.split():
                    ch_tokens_counter[token] += 1
        sorted_ch_list = sorted(ch_tokens_counter.items(), key=lambda kv: kv[1], reverse=True)
        with open(self._ch_vocab_path, 'w') as f:
            for k, v in sorted_ch_list[0: self._ch_limit]:
                f.write('%s\n' % (k))

        print('china tokens count is %d' % (len(ch_tokens_counter)))


if __name__=="__main__":
    data_path = os.path.join(root_path, file_name)
    en_corpus_path = os.path.join(root_path, en_corpus_name)
    ch_corpus_path = os.path.join(root_path, ch_corpus_name)
    en_vocab_path = os.path.join(root_path, en_vocab_name)
    ch_vocab_path = os.path.join(root_path, ch_vocab_name)
    dataset = T2TDataset(data_path, en_corpus_path, ch_corpus_path, en_vocab_path, ch_vocab_path)