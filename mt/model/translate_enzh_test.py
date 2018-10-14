from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import numpy as np

import tensorflow as tf


class TranslateTest(tf.test.TestCase):
    def testNp(self):
        num_units=5
        T=3
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

    def testSplit(self):
        T = 10
        N = 10
        num_units=10
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])
        print(position_enc)
        with tf.Session() as sess:
            print(sess.run(position_ind))

if __name__=='__main__':
    tf.test.main()