#! /usr/bin/python
# -*- coding:utf-8 -*-

import os,time
import numpy as np
import tensorflow as tf

from mt.dataset.mt_dataset import MTDataset

CH_VOCAB_SIZE=50000
EN_VOCAB_SIZE=8000
EMBEDDING_SIZE=100

dataset_path = '../data/train.txt'
preprocess_path = '../data/dataset.txt'
pickle_path = '../pkl/tokens.pkl'


class RNN_Seq2seq_model():
    def __init__(self):
        pass

    def model(self):
        ids=tf.constant(value=[[0, 5, 10, 1, 6, 11, 2, 3], [7, 12, 3, 2, 8, 4, 2, 9]], dtype=tf.int32)
        ch_vocab_embedding=tf.random_normal(mean=0, stddev=-1, shape=[CH_VOCAB_SIZE, EMBEDDING_SIZE], dtype=tf.float32, seed=7)

        value = tf.nn.embedding_lookup(params=ch_vocab_embedding, ids=ids, partition_strategy='mod')


if __name__=='__main__':
    dataset = MTDataset(dataset_path, 128, preprocess_path, en_vocab_size=50000, ch_vocab_size=8000,
                        pickle_path=pickle_path)
    ch_batch, en_batch = dataset.get_batch_op()
    print('call end')

    with tf.Session() as sess:
        dataset.init(sess=sess)
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for _ in range(1):
            batch1, batch2 = sess.run([ch_batch, en_batch])
            print(batch1)
            print(batch2)
        print('delay time: %d' % (time.time() - start_time))
    print('call end')