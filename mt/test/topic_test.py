import os, json

import tensorflow as tf

with tf.Session() as sess:
    print(sess.list_devices())

root_path = '/home/zhangpengpeng/Downloads'
file_name = '融合云sink配置列表json'

with tf.gfile.Open(os.path.join(root_path, file_name)) as f:
    content = f.read()
    print(len(content))
    topics = json.loads(content)
    for topic in topics:
        if topic['HdfsPath'] != '':
            print(topic['HdfsPath'] + '\t' +topic['Topic'])
