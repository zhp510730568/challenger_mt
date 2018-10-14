import os
import tensorflow as tf


data_path = '../data/train.txt'
base_path = '/home/zhangpengpeng/t2t_train/challenger_mt/'


def build_vocab():
    en_vocab=set()
    ch_vocab=set()
    with tf.gfile.Open(data_path, 'r') as f:
        for line in f:
            arr = line.split('\t')
            en_sen=arr[2].strip()
            zh_sen=arr[3].strip()
            for token in en_sen.split():
                en_vocab.add(token)
            ch_tokens = (token for token in zh_sen)
            for token in ch_tokens:
                ch_vocab.add(token)
        print(en_vocab)
        print(ch_vocab)

    with tf.gfile.Open(os.path.join(base_path, 'vocab.en'), 'w') as en_file:
        with tf.gfile.Open(os.path.join(base_path, 'vocab.ch'), 'w') as ch_file:
            for token in en_vocab:
                en_file.write('%s\n' % (token))
            for token in ch_vocab:
                ch_file.write('%s\n' % (token))


def split_dataset():
    with tf.gfile.Open(os.path.join(base_path, 'train.txt'), 'r') as file:
        with tf.gfile.Open(os.path.join(base_path, 'train_dataset.en.txt'), 'w') as train_en_file:
            with tf.gfile.Open(os.path.join(base_path, 'val_dataset.en.txt'), 'w') as val_en_file:
                with tf.gfile.Open(os.path.join(base_path, 'train_dataset.ch.txt'), 'w') as train_ch_file:
                    with tf.gfile.Open(os.path.join(base_path, 'val_dataset.ch.txt'), 'w') as val_ch_file:
                        count = 0
                        for sentence in file:
                            count += 1
                            if count % 50 != 0:
                                arr = sentence.split('\t')
                                train_en_file.write('%s\n' % (arr[2].strip()))
                                train_ch_file.write('%s\n' % (arr[3].strip()))
                            else:
                                arr = sentence.split('\t')
                                val_en_file.write('%s\n' % (arr[2].strip()))
                                val_ch_file.write('%s\n' % (arr[3].strip()))


if __name__=='__main__':
    tf.keras.layers.Conv1D