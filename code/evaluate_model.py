# -*- coding: utf8 -*-
from __future__ import unicode_literals
import sys, os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from modules.masked_cross_entropy import *
from modules.ControllableDataset import *
from modules.ControllableModel import *
from modules.AttentionDecoder import *
from modules.evaluate import *
from modules.MovingAvg import *
from modules.trainer import *
from scorer import *
import torch.utils.data as data_utils
import opts
import argparse
import cPickle
#import _pickle as cPickle
from modules.Vocab import Vocab

control_len = 1
gpu = True
bidirectional = True


def _cat_directions(h, bidirectional_encoder):
    if bidirectional_encoder:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h


def evaluate(encoder, decoder, src_sentence, control_vector, max_length, vocab_src, vocab_tgt, options):
    input_seqs, input_length = vocab_src.sentence_to_word_ids(src_sentence, max_length)
    input_lengths = [input_length]
    input_seqs = torch.LongTensor(input_seqs).unsqueeze(1)
    input_batches = Variable(input_seqs, volatile=True)

    # control_batches.unsqueeze(0)
    # print(control_batches.size())
    def get_control(s):
        # slist = s.split(' ')
        control_list = [float(x) for x in s]
        control_list = np.asarray(control_list, dtype='float32')
        control_tensor = torch.from_numpy(control_list)
        return control_tensor

    control_batches = get_control(control_vector)
    control_batches = control_batches.unsqueeze(0)
    control_batches = Variable(control_batches)

    if options.use_cuda and gpu:
        input_batches = input_batches.cuda()
        control_batches = control_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None, options)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([vocab_tgt.GO]), volatile=True)  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoder_hidden = _cat_directions(encoder_hidden, bidirectional)

    if options.use_cuda and gpu:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    if options.decoder != 'rnn':
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)


    #print(decoder_input)
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_output_projected, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, control_batches, options, False, True
        )
        if options.decoder != 'rnn':
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        decoder_output_projected = F.softmax(decoder_output_projected)
        topv, topi = decoder_output_projected.data.topk(1)
        ni = topi[0][0]
        if ni == vocab_tgt.EOS:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocab_tgt.index2word(ni))

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if options.use_cuda:
            decoder_input = decoder_input.cuda()
        # decoder_input = decoder_output_projected

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    if options.decoder != 'rnn':
        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]
    else:
        return decoded_words, None


def load_model(filename):
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        if gpu:
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        return checkpoint


def initialize_model(options, vocab_file, model_file):
    vocab_src = Vocab('model_vocab')
    vocab_src.load_vocab(vocab_file)
    vocab_tgt = vocab_src

    embedding = nn.Embedding(vocab_src.get_n_words, options.emb_size)
    encoder = EncoderRNN(vocab_src.get_n_words, options.rnn_size, options.emb_size, embedding, options.layers)
    encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())

    decoder_hidden_size = options.rnn_size
    if bidirectional:
        decoder_hidden_size = decoder_hidden_size * 2
    decoder = ControllableAttnDecoderRNN(decoder_hidden_size, vocab_tgt.get_n_words, options.emb_size,
                                         embedding, control_len, options.layers)


    if options.use_cuda and gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    checkpoint = load_model(model_file)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    return encoder, decoder, vocab_src, vocab_tgt


def test(options, vocab_file, model_file):
    encoder, decoder, vocab_src, vocab_tgt = initialize_model(options, vocab_file, model_file)
    while True:
        sentence = raw_input('Enter sentence: ')
        control_vec = raw_input('Enter control vector space separated of len ' + str(control_len) + ' : ')
        control_vec = control_vec.split()
        print(sentence, control_vec)
        decoded_words, attn = evaluate(encoder, decoder, sentence, control_vec, len(sentence.split()) + 1, vocab_src,
                                       vocab_tgt, options)
        print('Out ' + ' '.join(decoded_words))
        print('-----------')


def test_all(model_file, arg_file, vocab_file, sentences, scorer, outname):
    bname = os.path.basename(model_file)
    outname = outname + "_" + bname + '.csv'
    outfile = open(outname, 'w')
    control_vectors = [1, 2, 3, 4, 5]
    output_sentences = []
    rewards = []
    options = cPickle.load(open(arg_file, "rb"))
    print(options)
    encoder, decoder, vocab_src, vocab_tgt = initialize_model(options, vocab_file, model_file)
    for sentence in sentences:
        sentence = sentence.rstrip()
        o = {}
        fileoutstring = ''
        for control_vector in control_vectors:
            decoded_words, attn = evaluate(encoder, decoder, sentence, [control_vector], len(sentence.split()) + 1,
                                           vocab_src,
                                           vocab_tgt, options)
            generated =   ' '.join(decoded_words) + '\n'
            reward_package = all_rewards(sentence, generated, scorer, options)
            reward = reward_package[0]
            fileoutstring = fileoutstring + generated + '\t' + str(reward) + '\t'
            o[control_vector] = generated + ' ##Reward## ' + str(reward)
        output_sentences.append(o)
        outfile.write(sentence + fileoutstring + '\n')
    outfile.close()
    return output_sentences




def test_all_sentences():
    model_file = sys.argv[1]
    arg_file = sys.argv[2]
    vocab_file = sys.argv[3]
    sentence_file = sys.argv[4]
    sentences = open(sentence_file, 'r').readlines()
    options = cPickle.load(open(arg_file, "rb"))
    print(options)
    outfile = open('outfile_evaluate_all.txt', 'w')
    encoder, decoder, vocab_src, vocab_tgt = initialize_model(options, vocab_file, model_file)
    control_vectors = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for sentence in sentences:
        sentence = sentence.rstrip()
        outfile.write(sentence + '\n')
        for control_vector in control_vectors:
            decoded_words, attn = evaluate(encoder, decoder, sentence, [control_vector], len(sentence.split()) + 1,
                                           vocab_src,
                                           vocab_tgt, options)
            outfile.write('Output::  ' + ' '.join(decoded_words) + '\n')
        outfile.write('-----------' + '\n')






def test_single():
    model_file = sys.argv[1]
    arg_file = sys.argv[2]
    vocab_file = sys.argv[3]
    options = cPickle.load(open(arg_file, "rb"))
    print(options)
    test(options, vocab_file, model_file)

def main():
    test_all_sentences()

if __name__ == '__main__':
    main()
