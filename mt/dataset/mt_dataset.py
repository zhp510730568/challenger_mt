#! /usr/bin/python

import os, json, time

import numpy as np

import nltk

import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.keras.preprocessing import sequence

from mt.dataset.dictionary import Vocabulary


class MTDataset(object):
    '''
    train dataset
    '''
    def __init__(self, dataset_path, batch_size, preprocess_path=None, en_vocab_size=10000, ch_vocab_size=10000, pickle_path=None):
        self._dataset_path=dataset_path
        if preprocess_path is None:
            raise ValueError('preprocess_path must not be None')
        self._preprocess_path = preprocess_path
        if not isinstance(batch_size, int) or batch_size < 0 or batch_size > 100000:
            raise ValueError('batch_size must be int type, and must be between 0 and 100000')
        self._batch_size=batch_size
        self._en_vocab_size=en_vocab_size
        self._ch_vocab_size=ch_vocab_size
        self._pickle_path=pickle_path
        self._is_init=False
        self._vocabulary=Vocabulary(self._dataset_path, self._en_vocab_size, self._ch_vocab_size, self._pickle_path)
        if not os.path.exists(self._preprocess_path):
            self._preprocess()

    def init(self, sess=None):
        if self._is_init:
            return
        if sess is None or (not isinstance(sess, Session)):
            raise ValueError('sess type is not correct')
        file_dataset = tf.data.TextLineDataset(self._preprocess_path)
        self.file_dataset = file_dataset.shuffle(buffer_size=100) \
            .repeat() \
            .map(map_func=lambda sentence: tf.py_func(map_fn, [sentence], [tf.int32, tf.int32]), num_parallel_calls=6) \
            .batch(self._batch_size) \
            .prefetch(buffer_size=self._batch_size * 10)
        self.init_op = self.file_dataset.make_initializable_iterator(shared_name='train_dataset')
        sess.run(self.init_op.initializer)
        self._is_init=True

    def get_batch_op(self):
        if not self._is_init:
            raise ValueError('init must be called first')

        ch_batch, en_batch = self.init_op.get_next()

        return ch_batch, en_batch

    def _preprocess(self):
        count=0
        with open(self._dataset_path, 'r') as train_file, open(self._preprocess_path, 'w') as dataset_file:
            for sentence in train_file:
                arr = sentence.strip().split('\t')
                DocID = arr[0]
                SenID = arr[1]
                EngSen = arr[2].strip()
                ChnSen = arr[3].strip()
                count += 1
                en_ids = self._vocabulary.en_doc_to_id(nltk.word_tokenize(EngSen))
                ch_ids = self._vocabulary.ch_doc_to_id(ChnSen)
                info = {'DocID': DocID, 'SenID': SenID, 'EngSen': en_ids, 'ChnSen': ch_ids}
                dataset_file.write('%s\n' % (json.dumps(info)))
                if count % 100000 == 0:
                    print('processed data1: %d\t%s' % (
                    count, time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))))


def map_fn(sentence):
    m = json.loads(sentence.decode(encoding = "utf-8"))
    arr =sequence.pad_sequences(np.array([m['ChnSen']]), maxlen=100, padding='post')
    arr1 = sequence.pad_sequences(np.array([m['EngSen']]), maxlen=100, padding='post')

    return arr[0].astype(np.int32), arr1[0].astype(np.int32)


if __name__=='__main__':
    dataset_path = '../data1/train.txt'
    preprocess_path='../data1/dataset.txt'
    pickle_path = '../pkl/tokens.pkl'

    with Session() as sess:
        dataset = MTDataset(dataset_path, 128, preprocess_path, en_vocab_size=50000, ch_vocab_size=8000, pickle_path=pickle_path)
        dataset.init(sess=sess)
        ch_batch, en_batch=dataset.get_batch_op()

        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for _ in range(1):
            batch1, batch2 = sess.run([ch_batch, en_batch])
            print(batch1)
            print(batch2)
        print('delay time: %d' % (time.time() - start_time))
    print('call end')