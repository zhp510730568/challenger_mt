import numpy as np

import tensorflow as tf

BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10

with tf.device("/GPU:0"):
    var = tf.get_variable(shape=[5000, 5000], dtype=tf.float32, name='var', initializer=tf.orthogonal_initializer(gain=1.0, seed=7))

arr = tf.unstack(var, axis=0)

result = arr[0] * arr[1]

tf.erf
x_shape=[1, 2,3, 4]
print([-1] + x_shape[-3:])