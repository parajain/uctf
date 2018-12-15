from math import log
from numpy import array
from numpy import argmax
import torch
from torch.nn import functional
from torch.autograd import Variable
from Vocab import *

# beam search
def beam_search(data, k):
    data = data.data.numpy()
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def beam_search_decoder(data, k, vocab):
    print('Data ', data.size())
    result = beam_search(data, k)
    for r in result:
        seq = r[0]
        score = r[1]
        for s in seq:
            print(vocab.index2word(s))
        print(seq)
        print(score)



def test():
    max_len = 5
    vocab_file = 'data/vocab_mc5.txt'
    vocab_src = Vocab('model_vocab')
    vocab_src.load_vocab(vocab_file)
    vocab_size = vocab_src.get_n_words
    outputs = torch.randn(max_len, vocab_size)
    outputs = functional.softmax(outputs)
    beam_search_decoder(outputs, 3, vocab_src)



def main():
    # define a sequence of 10 words over a vocab of 5 words
    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1]]
    data = array(data)
    # decode sequence
    beam_search_decoder(data, 3)

if __name__ == '__main__':
    test()
