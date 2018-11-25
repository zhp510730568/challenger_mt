from __future__ import print_function

import csv
import tensorflow as tf

input_file = "./data1/glue_data/MRPC/train.tsv"

vars = tf.train.list_variables("/home/zhangpengpeng/PycharmProjects/challenger_mt/challenger/model/model.ckpt-1036825")
count = 0
for var in vars:
    print(var)
    if len(var[1])==2:
        count += var[1][0] * var[1][1]
    elif len(var[1]) == 1:
        count += var[1][0]

print(count)