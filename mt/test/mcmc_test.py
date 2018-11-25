import tensorflow as tf
import torch
init = tf.constant(value=[[0.3, 0.3, 0.4]])

transfer = tf.constant(value=[[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])

result = init
with tf.device("/GPU:0"):
    for _ in range(20000):
        result = tf.matmul(result, transfer)

sum = tf.reduce_sum(result)

with tf.Session() as sess:
    print(sess.run(result))
    print(sess.run(sum))