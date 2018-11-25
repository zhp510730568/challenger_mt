#! /usr/bin/python
# -*- coding:utf-8 -*-

import os,time
import numpy as np
import tensorflow as tf

from mt.dataset.mt_dataset import MTDataset

EN_VOCAB_SIZE=50000
CH_VOCAB_SIZE=8000
EMBEDDING_SIZE=256
HIDDEN_SIZE=256

batch_size=256

dataset_path = '../data1/train.txt'
preprocess_path = '../data1/dataset.txt'
pickle_path = '../pkl/tokens.pkl'

device_spec = tf.DeviceSpec(device_type='CPU', device_index=0)


class RNN_Seq2seq_model:
    def __init__(self):
        pass

    def model(self, ch_batch, en_batch, sess):
        ch_vocab_embedding=tf.random_normal(mean=0, stddev=-1, shape=[CH_VOCAB_SIZE, EMBEDDING_SIZE],
                                            dtype=tf.float32, seed=5)
        en_vocab_embedding=tf.random_normal(mean=0, stddev=-1, shape=[EN_VOCAB_SIZE, EMBEDDING_SIZE],
                                              dtype=tf.float32, seed=7)
        en_value = tf.nn.embedding_lookup(params=en_vocab_embedding, ids=en_batch, partition_strategy='mod')
        ch_value = tf.nn.embedding_lookup(params=ch_vocab_embedding, ids=ch_batch, partition_strategy='mod')
        rnn_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)

        outputs, state = tf.nn.dynamic_rnn(rnn_cell, ch_value, dtype=tf.float32)
        return outputs, state


if __name__=='__main__':
    rnn_model = RNN_Seq2seq_model()
    with tf.Session() as sess:
        dataset = MTDataset(dataset_path, batch_size, preprocess_path, EN_VOCAB_SIZE, CH_VOCAB_SIZE, pickle_path)
        dataset.init(sess=sess)
        ch_batch, en_batch = dataset.get_batch_op()
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for _ in range(1):
            ch_batch_data, en_batch_data = sess.run([ch_batch, en_batch])
            print(sess.run(rnn_model.model(ch_batch_data, en_batch_data, sess)))
        print('delay time: %d' % (time.time() - start_time))
    print('call end')
    tf.layers.conv1d