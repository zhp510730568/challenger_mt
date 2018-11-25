from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

UNK='<UNK>'


def generate_lines_for_vocab(data_dir, source):
    filepath = os.path.join(data_dir, source[0])
    with tf.gfile.GFile(filepath, mode="r") as source_file:
        for line in source_file:
            yield line


def get_or_generate_vocab(data_dir, vocab_filename, vocab_size, source):
  vocab_generator = generate_lines_for_vocab(data_dir, source)
  return generator_utils.get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     vocab_generator)


@registry.register_problem
class TranslateEnzhToken(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def approx_vocab_size(self):
    return 2**16  # 32k

  @property
  def oov_token(self):
    return "UNK"

  @property
  def is_generate_per_split(self):
      return False

  @property
  def vocab_filename(self):
      return 'en_vocab'

  @property
  def source_vocab_name(self):
      return "%s.en" % self.vocab_filename

  @property
  def target_vocab_name(self):
      return "%s.zh" % self.vocab_filename

  @property
  def dataset_splits(self):
      return [{
          "split": problem.DatasetSplit.TRAIN,
          "shards": 100,
      }, {
          "split": problem.DatasetSplit.EVAL,
          "shards": 1,
      }]

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    source_dataset = ['en_corpus']
    target_dataset = ['ch_corpus']
    source_vocab = get_or_generate_vocab(
        data_dir,
        self.source_vocab_name,
        self.approx_vocab_size,
        source_dataset)

    target_vocab = get_or_generate_vocab(
        data_dir,
        self.target_vocab_name,
        self.approx_vocab_size,
        target_dataset)

    tag = "train" if train else "dev"
    filename_base = "challenger_enzh_%sk_tok_%s" % (self.approx_vocab_size, tag)

    return text_problems.text2text_generate_encoded(
        text_problems.text2text_txt_iterator(os.path.join(data_dir, 'en_corpus'),
                                             os.path.join(data_dir, 'ch_corpus')),
        source_vocab, target_vocab)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


if __name__=='__main__':
    root_path='/home/zhangpengpeng/PycharmProjects/challenger_mt/mt/data1'
    source_vocab = text_encoder.TokenTextEncoder(os.path.join(root_path, 'en_vocab'), num_reserved_ids=2, replace_oov=UNK)
    target_vocab = text_encoder.TokenTextEncoder(os.path.join(root_path, 'ch_vocab'), num_reserved_ids=2, replace_oov=UNK)
    encoded = text_problems.text2text_generate_encoded(
        text_problems.text2text_txt_iterator(os.path.join(root_path, 'en_corpus'),
                                             os.path.join(root_path, 'ch_corpus')),
        source_vocab, target_vocab)
    for _ in range(100):
        print(encoded())
