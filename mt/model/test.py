#! /usr/bin/python
import numpy as np
import tensorflow as tf


def generate_input(lower_length=10, upper_length=100, vocab_size=1000, batch_size=10):
    def random_length():
        if lower_length <= 0:
            raise ValueError('lower_length must be greater than 0')
        if lower_length > upper_length:
            return lower_length
        return np.random.randint(low=lower_length, high=upper_length+1, size=1, dtype=np.int32)

    while True:
        for _ in range(100):
            print(random_length())
        yield [np.random.randint(low=0, high=vocab_size, size=random_length()).tolist() for _ in range(batch_size)]


data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
x = tf.strided_slice(data,[0,0,0],[1,1,1])
y = tf.strided_slice(data,[0,0,0],[2,2,2],[1,1,1])
z = tf.strided_slice(data,[0,0,0],[2,2,2],[1,2,1])

with tf.Session() as sess:
    print('x', sess.run(x))
    print('y', sess.run(y))
    print('z', sess.run(z))