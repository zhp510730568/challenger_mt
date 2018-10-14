import tensorflow as tf
from tensorflow.python.ops.array_ops import reverse_sequence
from tensorflow.python.util import nest

from tensorflow.python.framework import ops
from tensorflow.python.ops.array_ops import rank
from tensorflow.python.ops import array_ops


i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])

range = tf.range(20)
range = tf.reshape(range, shape=[2, 2, 5])

seqence_lenght = tf.constant(value=[5, 5])

reverse_range = reverse_sequence(range, seq_lengths=seqence_lenght, seq_dim=2, batch_dim=0)

splits = array_ops.split(
        value=range, num_or_size_splits=2, axis=1)

with tf.Session() as sess:
    print('source data: ', sess.run(range))
    print('split: ', sess.run(splits))