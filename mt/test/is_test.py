import numpy as np
import tensorflow as tf

global_step = tf.get_variable(name="global_step", dtype=tf.int64, shape=(), trainable=False, initializer=tf.zeros_initializer)
assign_op = tf.assign(global_step, 20)
with tf.control_dependencies([assign_op]):
    decalyed_learning_rate = tf.train.polynomial_decay(learning_rate=0.1, global_step=global_step, decay_steps=20)

is_warmup = tf.cast(10 < 100, tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(global_step))
    print(sess.run(decalyed_learning_rate))
    print(sess.run(global_step))
    print(sess.run(is_warmup))
