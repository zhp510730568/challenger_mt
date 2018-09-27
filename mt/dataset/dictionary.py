#! /usr/bin/python
# -*- coding:utf-8 -*-
import os, pickle, nltk, json

from collections import Counter, Iterable

# unkown token
UNKOWN = '<unk>'
# start of sentence
SOS = '<sos>'
# end of sentence
EOS = '<eos>'


class Vocabulary(object):
    def __init__(self, train_path, en_vocab_size=10000, ch_vocab_size=10000, pickle_path=None):
        self._train_path = train_path
        self._pickle_path = pickle_path
        if self._train_path is None or not os.path.exists(self._train_path):
            raise ValueError("train data path is None or don't exist")

        if self._pickle_path is None:
            raise ValueError('pickle path is None')

        self._en_vocab_size = en_vocab_size
        self._ch_vocab_size = ch_vocab_size

        if os.path.exists(self._pickle_path):
            with open(self._pickle_path, 'br') as f:
                d = pickle.load(f)
                self._en_counter = d['en_tokens']
                self._ch_counter = d['ch_tokens']
        else:
            self._load()
        self._compute_vocab()

    def _load(self):
        self._en_counter = Counter()
        self._ch_counter = Counter()
        with open(train_path, 'r') as f:
            count = 0
            for sentence in f:
                arr = sentence.split('\t')
                count += 1
                en_tokens = self.en_tokenize(arr[2].strip())
                ch_tokens = self.ch_tokenize(arr[3].strip())

                for token in en_tokens:
                    self._en_counter[token] += 1
                for token in ch_tokens:
                    self._ch_counter[token] += 1
        m = {'en_tokens': self._en_counter, 'ch_tokens': self._ch_counter}
        with open(self._pickle_path, 'bw') as f:
            pickle.dump(m, f)

    def _compute_vocab(self):
        sorted_ch_list = sorted(self._ch_counter.items(), key=lambda kv: kv[1], reverse=True)

        ch_vocab = [token for token, _ in sorted_ch_list[0: self._ch_vocab_size - 3]]
        self._ch_word_to_id = {token: index + 3 for index, token in enumerate(ch_vocab)}
        self._ch_word_to_id[SOS] = 0
        self._ch_word_to_id[UNKOWN] = 1
        self._ch_word_to_id[EOS] = 2

        self._ch_id_to_word = {index + 3: token for index, token in enumerate(ch_vocab)}
        self._ch_id_to_word[0] = SOS
        self._ch_id_to_word[1] = UNKOWN
        self._ch_id_to_word[2] = EOS
        print(self._ch_word_to_id)

        sorted_en_list = sorted(self._en_counter.items(), key=lambda kv: kv[1], reverse=True)
        en_vocab = [token for token, _ in sorted_en_list[0: self._en_vocab_size - 3]]
        self._en_word_to_id = {token: index + 3 for index, token in enumerate(en_vocab)}
        self._en_word_to_id[SOS] = 0
        self._en_word_to_id[UNKOWN] = 1
        self._en_word_to_id[EOS] = 2
        print(self._en_word_to_id)

        self._en_id_to_word = {index + 3: token for index, token in enumerate(en_vocab)}
        self._en_id_to_word[0] = SOS
        self._en_id_to_word[1] = UNKOWN
        self._en_id_to_word[2] = EOS

    @staticmethod
    def en_tokenize(sentence):
        if not isinstance(sentence, str):
            raise ValueError('sentence must be str type')
        return nltk.word_tokenize(sentence, language='english')

    @staticmethod
    def ch_tokenize(sentence):
        if not isinstance(sentence, str):
            raise ValueError('sentence must be str type')
        return [token for token in sentence]

    def en_doc_to_id(self, doc):
        '''
        convert doc to ids
        :param time_steps:
        :param doc:
        :return:
        '''
        if isinstance(doc, Iterable):
            ids = []
            ids.append(0)
            for token in doc:
                if token in self._en_word_to_id:
                    ids.append(self._en_word_to_id[token])
                else:
                    ids.append(1)
            ids.append(2)
            return ids
        else:
            raise ValueError('doc must be str type')

    def en_id_to_doc(self, ids):
        '''
        convert doc to ids
        :param ids:
        :return:
        '''
        if isinstance(ids, Iterable):
            tokens = []
            for id in ids:
                if id in self._en_id_to_word:
                    tokens.append(self._en_id_to_word[id])
                else:
                    tokens.append(UNKOWN)  # UNKOWN
            return tokens
        else:
            raise ValueError('doc must be str type')

    def ch_doc_to_id(self, doc):
        '''
        convert doc to ids
        :param time_steps:
        :param doc:
        :return:
        '''
        if isinstance(doc, Iterable):
            ids = []
            ids.append(0)
            for token in doc:
                if token in self._ch_word_to_id:
                    ids.append(self._ch_word_to_id[token])
                else:
                    ids.append(1)
            ids.append(2)
            return ids
        else:
            raise ValueError('doc must be str type')

    def ch_id_to_doc(self, ids):
        '''
        convert doc to ids
        :param ids:
        :return:
        '''
        if isinstance(ids, Iterable):
            tokens = []
            for id in ids:
                if id in self._ch_id_to_word:
                    tokens.append(self._ch_id_to_word[id])
                else:
                    tokens.append(UNKOWN)  # UNKOWN
            return tokens
        else:
            raise ValueError('doc must be str type')


if __name__ == '__main__':
    import time
    train_path = '../data/train.txt'
    pickle_path = '../pkl/tokens.pkl'
    vocab = Vocabulary(train_path=train_path, en_vocab_size=50000, ch_vocab_size=8000, pickle_path=pickle_path)
    start_time=time.time()
    count = 0

    with open(train_path, 'r') as train_file, open('../data/dataset.txt', 'w') as dataset_file:
        for sentence in open(train_path):
            arr = sentence.strip().split('\t')
            DocID=arr[0]
            SenID=arr[1]
            EngSen=arr[2].strip()
            ChnSen=arr[3].strip()
            count+=1
            en_ids = vocab.en_doc_to_id(nltk.word_tokenize(EngSen))
            EngSen=vocab.en_id_to_doc(en_ids)
            ch_ids = vocab.ch_doc_to_id(ChnSen)
            print('input: ', ChnSen)
            print('ch_ids: ', ch_ids)
            print('ch_sen: ', vocab.ch_id_to_doc(ch_ids))
            info = {'DocID': DocID, 'SenID': SenID, 'EngSen': en_ids, 'ChnSen': ch_ids}
            dataset_file.write('%s\n' % (json.dumps(info)))
            if count % 100000 == 0:
                print('processed data: %d\t%s' % (count, time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time()))))