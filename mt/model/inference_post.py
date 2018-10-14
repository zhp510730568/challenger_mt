import os

import numpy as np

import tensorflow as tf

root_path = '../output1/decode'
decode_name = 'translation_900000.zh'


def merge_sentence():
    data_path = os.path.join(root_path, decode_name)
    with tf.gfile.Open(data_path) as f:
        with tf.gfile.Open(os.path.join(root_path, 'decode_output1.sgm'), 'w') as decode_file:
            for sentence in f.readlines():
                sentence = np.array([token for token in sentence.strip()])
                sentence[1::2] = ''
                decode_file.write('%s\n' % (''.join(sentence)))


if __name__=='__main__':
    merge_sentence()