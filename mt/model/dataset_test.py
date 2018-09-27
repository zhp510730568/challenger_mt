#! /usr/bin/python
import os
import numpy as np
import tensorflow as tf

params=tf.constant(value=[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 2, 8], [4, 2, 9]], dtype=tf.float32)
ids=tf.constant(value=[1], dtype=tf.float32)

value = tf.nn.embedding_lookup(params=params, ids=6, partition_strategy='mod')

with tf.Session() as sess:
    result = sess.run(value)
    print(result)