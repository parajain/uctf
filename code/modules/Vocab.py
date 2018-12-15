#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
tools to build vocabulary
"""

from nltk.tokenize import word_tokenize
from collections import Counter
from modules.PretrainedEmbeddings import *
import numpy as np

# from clean_str import *


class Vocab(object):

    PAD = 0
    EOS = 1
    OOV = 2
    GO = 3

    def __init__(self, vocab_name):
        self.word_to_id = None
        self.id_to_word = None
        self.vocab_name = vocab_name

    def load_vocab(self, word_to_id_file):
        f = open(word_to_id_file, 'r')
        self.word_to_id = eval(f.read())
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))
        self.OOV = self.word_to_id['<OOV>']
        self.PAD = self.word_to_id['<PAD>']
        self.GO = self.word_to_id['<GO>']
        self.EOS = self.word_to_id['<EOS>']

    def build_vocab(self, files, out_file_name, min_count=0, tokenizer='simple'):
        sentences = []
        for file in files:
            f = open(file, 'r')
            lines = f.readlines()
            sentences.extend(lines)

        self.word_to_id, self.id_to_word = self._build_vocabulary(sentences, min_count, tokenizer)
        self.save_vocab(out_file_name)

    def simple_tokenize(self, string):
        return string.strip().split(' ')

    def sentence_to_word_ids(self, sentence, max_sequence_length=None, tokenizer='simple', prependGO=False, eos = True, sos=False):
        """
        encode a given [sentence] to a list of word ids using the vocabulary dict [word_to_id]
        adds a end-of-sentence marker (<EOS>)
        out-of-vocabulary words are mapped to 2
        """
        if prependGO:
            tokens = ['<GO>']
        else:
            tokens = []
        if tokenizer == 'nltk':
            tokens.extend(word_tokenize(sentence))
        else:
            tokens.extend(self.simple_tokenize(sentence))

        tokens.append('<EOS>')
        tokens_length = len(tokens)
        if max_sequence_length is not None:
            if len(tokens) <= max_sequence_length:
                tokens += ['<PAD>' for i in range(max_sequence_length - len(tokens))]
            else:
                tokens = tokens[:max_sequence_length]
        tokens_length = min(tokens_length, max_sequence_length)
        return [self.word_to_id.get(word, self.OOV) for word in tokens], tokens_length

    def word_ids_to_sentence(self, word_ids_list):
        """ decode a given list of word ids [word_ids_list] to a sentence using the inverse vocabulary dict [id_to_word]
        """
        tokens = [self.id_to_word.get(id) for id in word_ids_list if id >= 2]

        return ' '.join(tokens).capitalize() + '.'

    def _build_vocabulary(self, sentences, min_count, tokenizer):
        """ build the vocabulary from a list of `sentences'
        uses word_tokenize from nltk for word tokenization

        :params:
            sentences: list of strings
                the list of sentences
            min_count: int
                keep words whose count is >= min_count

        :returns:
           word_to_id: dict
                dict mapping a word to its id, e.g., word_to_id['the'] = 4
                the id start from 4
                3 is reserved for <GO> (in case of decoder RNN for En-Dec architecture)
                2 is reserved for out-of-vocabulary words (<OOV>)
                1 is reserved for end-of-sentence marker (<EOS>)
                0 is reserved for padding (<PAD>)
        """
        print('#################### Building vocabulary ##########################')
        wordcount = Counter()
        for sentence in sentences:
            if tokenizer == 'nltk':
                tokens = word_tokenize(sentence)
            else:
                tokens = self.simple_tokenize(sentence)
            wordcount.update(tokens)

        print('vocabulary size = %d' % (len(wordcount)))

        # filtering
        count_pairs = wordcount.most_common()
        count_pairs = [c for c in count_pairs if c[1] >= min_count]

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(4, len(words) + 4)))
        print('vocabulary size = %d (after filtering with min_count =  %d)' % (len(word_to_id), min_count))

        word_to_id['<PAD>'] = self.PAD
        word_to_id['<EOS>'] = self.EOS
        word_to_id['<OOV>'] = self.OOV
        word_to_id['<GO>'] = self.GO

        id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

        return word_to_id, id_to_word
    def word2index(self,w):
        try:
            idx = self.word_to_id[w]
        except:
            idx = self.OOV
        return idx
    def index2word(self, idx):
        return self.id_to_word[idx]

    def save_vocab(self, filename):
        f = open(filename, 'w')
        f.write(str(self.word_to_id))

    def print_vocab(self):
        print(self.word_to_id)

    @property
    def get_n_words(self):
        return len(self.word_to_id)

    def get_glove_embeddings(self, filename, embedding_size):
        weights = load_glove_embedding(filename, embedding_size, self.id_to_word, self.word_to_id)
        return weights


