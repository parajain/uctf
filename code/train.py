import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
import sys,shutil, os
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
from modules.trainer import *
from scorer import *
import torch.utils.data as data_utils
import opts
import argparse
import cPickle
from modules.Vocab import Vocab
from vectorizer import *
from score_model import *


control_len = 1 # We can have more than one controls
bidirectional = True

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    print('Saving model at ' + filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_model(encoder, decoder, vocab_src, vocab_tgt, encoder_optimizer, epoch, options):
    #TODO: Save vocabulary object
    #vocab_src.save_vocab(options.save_model + ".vocab")
    model = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'start_epoch':epoch,
        'options':options
    }
    save_checkpoint(model, False, options.save_model + '_epoch_' + str(epoch))


def save_pretrained(encoder, decoder, options):
    print('#### saving pretrained encoder #####')
    model = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }
    save_checkpoint(model, False, options.save_pre_trained_filename)


def load_model(options):
    if options.resume:
        if os.path.isfile(options.resume):
            print("=> loading checkpoint '{}'".format(options.resume))
            checkpoint = torch.load(options.resume)
            return checkpoint


def pretrain(dataloader, encoder, pre_train_decoder,optimizer, vocab_tgt, options):
    print('########## pretraining #############')
    epoch = 0
    while epoch < options.num_epoch_pretrain:
        epoch = epoch + 1
        epoch_loss = 0
        for sample_batched in dataloader:
            input_batches = Variable(sample_batched['src']).transpose(0, 1)  # will give seq_len x batch_size
            control_batches = Variable(sample_batched['control_tensor'])# this is  bs x control_len

            input_lengths = sample_batched['src_len']
            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_batches = input_batches[:, perm_idx]
            input_lengths = [x for x in input_lengths]
            control_batches = control_batches[perm_idx, :]

            loss = train_batch(
                input_batches, input_lengths, input_batches, input_lengths,
                control_batches, encoder, pre_train_decoder,
                optimizer, options, vocab_tgt
            )
            epoch_loss = epoch_loss + loss

        print('Epoch loss ', epoch_loss)

    save_pretrained(encoder, pre_train_decoder, options)


def initialize_model(options):
    vocab_src = Vocab('model_vocab')
    vocab_src.load_vocab(options.vocab_file)
    vocab_tgt = vocab_src

    print('Number of words in source: ', vocab_src.get_n_words)
    print('Number of words in target: ', vocab_tgt.get_n_words)

    # Initialize models
    embedding = nn.Embedding(vocab_src.get_n_words, options.emb_size)
    encoder = EncoderRNN(vocab_src.get_n_words, options.rnn_size, options.emb_size, embedding, options.layers,
                         0, bidirectional=bidirectional)
    encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())

    decoder_hidden_size = options.rnn_size
    if bidirectional:
        decoder_hidden_size = decoder_hidden_size * 2

    decoder = ControllableAttnDecoderRNN(decoder_hidden_size, vocab_tgt.get_n_words, options.emb_size,
                                         embedding, control_len, options.layers)


    pre_train_decoder = decoder
    #if True:#options.pretrain_decoder:
    #    pre_train_decoder = decoder
    #else:
    #    pre_train_decoder = ControllableDecoder(options.rnn_size, vocab_tgt.get_n_words,options.emb_size,  embedding, options.layers)


    if options.use_cuda:
        decoder.cuda()
        encoder.cuda()
        embedding.cuda()
        pre_train_decoder.cuda()

    opt_param_pretrain = (
        set(encoder_parameters) |
        set(pre_train_decoder.parameters()) |
        set(embedding.parameters()))

    pretrain_optimizer = optim.Adam(opt_param_pretrain, lr=options.learning_rate,eps=1e-3, amsgrad=True)
    #pretrain_optimizer = optim.Adam(opt_param_pretrain, lr=options.learning_rate)

    opt_param = (
        set(encoder_parameters) |
        set(decoder.parameters()) |
        set(embedding.parameters()))
    optimizer = optim.Adam(opt_param, lr=options.learning_rate,eps=1e-3, amsgrad=True)
    #optimizer = optim.Adam(opt_param, lr=options.learning_rate)

    if options.resume:
        print('Resuming')
        checkpoint = load_model(options)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        options.start_epoch = checkpoint['start_epoch']
    else:
        if not options.pretrain and options.load_pretrained:
            if os.path.isfile(options.pre_trained_model):
                print('########WARNING IT ONLY LOADS PRETRAINED DOES NOT RESUME A TRAINED MODEL: Loading pretrained model #######')
                checkpoint = torch.load(options.pre_trained_model)
                encoder.load_state_dict(checkpoint['encoder'])
                pre_train_decoder.load_state_dict(checkpoint['decoder'])
                decoder.load_state_dict(checkpoint['decoder'])
        if options.embeddings:
            print('######## Loading embeddings from, ', options.embeddings)
            pretrained_weight = vocab_src.get_glove_embeddings(options.embeddings, options.emb_size)
            #encoder.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            if not options.train_embedding:
                embedding.weight.requires_grad = False

    return vocab_src, vocab_tgt, encoder, decoder,pre_train_decoder, optimizer,pretrain_optimizer


def train(options):
    vocab_src, vocab_tgt, encoder, decoder, pre_train_decoder, encoder_optimizer,pretrain_optimizer = initialize_model(options)
    scorer = Scorer(options)
    word_to_id = vocab_src.word_to_id
    simmat = None
    vectorizer_fn = None
    syn_function = cPickle.load(open("syn.dat",'rb'))
    '''
    if options.use_vector:
        syn_resources = cPickle.load(open("syn.dat",'rb'))
        #if os.path.exists('simmat.dat'):
        #    with open("simmat.dat", "rb") as f:
        #        simmat = cPickle.load(f)
        #else:
        #    simmat = pre_compute_similarity_matrix(word_to_id)
        #    cPickle.dump(simmat, open("simmat.dat", "wb"))
        #vectorizer_fn = vector_builder_for_ce
        syn_function = syn_resources
    '''
    smodel, scriterion, soptimizer = init_score_model(options.vocab_file)
    if options.use_cuda:
        smodel = smodel.cuda()

    if options.pretrain:
        if not os.path.isfile(options.pretrain_src):
            print('ERROR: Source file to pretrain is not passes. Use -pretrain_src')

        pretrain_lines = open(options.pretrain_src, 'r').readlines()
        pretrain_dataset = ControllableUnsupervisedDatasetFromArray(pretrain_lines, None, control_len, "pretrain_data", vocab_src,
                                                                options.src_seq_length)
        pretrain_dataloader = data_utils.DataLoader(pretrain_dataset, batch_size=options.batch_size,
                                                    shuffle=options.shuffle, num_workers=1)
        pretrain(pretrain_dataloader, encoder, pre_train_decoder, pretrain_optimizer, vocab_tgt, options)
        print('####################### Pretraining Ends here')
        #generate(encoder, pre_train_decoder, valid_inputs, options, vocab_src, vocab_tgt, 0, True)

    epoch = options.start_epoch
    avg_reward = 0
    total_reward = 0
    train_with_rl = True
    count_train_without_rl = 0
    sampled = set(["1"]) #will contain sampled output
    prev_total_reward = 0
    src_lines = open(options.train_src, 'r').readlines()
    valid_lines = open(options.valid_src, 'r').readlines()
    explore_dataset = ControllableUnsupervisedDatasetFromArray(src_lines, None, control_len, "pretrain_data",
                                                                vocab_src,
                                                                options.src_seq_length)
    explore_dataloader = data_utils.DataLoader(explore_dataset, batch_size=options.batch_size,
                                                shuffle=options.shuffle, num_workers=1)

    print('######### Training ########')
    while epoch < options.epochs:
        epoch += 1
        if epoch%5 == 0:
            print("Training epoch ", epoch)
        print ("Exploring ", epoch)
        batch_num = 0
        total_reward = 0
        
        #hold the new data (sampled) 
        updated_src = []
        updated_tgt = []
        updated_control = []
        for sample_batched in explore_dataloader:
            batch_num = batch_num + 1
            print('batch_num ', batch_num)
            input_batches = Variable(sample_batched['src']).transpose(0, 1) #will give seq_len x batch_size
            control_batches = Variable(sample_batched['control_tensor'])  # this is  bs x control_len

            input_lengths = sample_batched['src_len']

            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_batches = input_batches[:, perm_idx]
            input_lengths = [x for x in input_lengths]
            control_batches = control_batches[perm_idx, :]

            new_src, new_tgt, new_control, curr_avg_reward = train_batch_rl_tf(
                input_batches, input_lengths, control_batches,
                encoder, decoder, encoder_optimizer, options, vocab_tgt,
                avg_reward, scorer, epoch, options.use_vector, syn_function,sampled)
            total_reward = total_reward + curr_avg_reward

            updated_src = updated_src + new_src
            updated_tgt = updated_tgt + new_tgt
            updated_control = updated_control + new_control

            try:
                sys.stdout.flush()
            except:
                pass

        print('length of new updated samples ', len(updated_src))
        print ("Sampled count ", len(sampled))
        print('************************ av ', total_reward)
        if len(updated_src) < options.batch_size - 2:
            print('Continuing since less data sampled ' + str(len(updated_src)))

        print('##############Training reward predictor model#############')
        smodel.train(True)
        train_score_model(updated_src, updated_tgt, updated_control, vocab_src, vocab_tgt, smodel, scriterion, soptimizer, options)

        
        #if total_reward<prev_total_reward:
        #    print "Batch reward did not improve. Skipping Exploitation"
        #    continue
        prev_total_reward = total_reward
        datasetPre = ControllableSupervisedDatasetFromArray(updated_src,updated_tgt, updated_control, 'unit_test',
                                                            vocab_src , vocab_tgt,options.src_seq_length,options.src_seq_length)
        num_instancesPre = datasetPre.num_instances
        num_batchesPre = math.ceil(num_instancesPre / options.batch_size)
        dataloaderPre = data_utils.DataLoader(datasetPre, batch_size=options.batch_size, shuffle=options.shuffle, num_workers=1)
        
        print ("Exploiting ", epoch)
        ex_epoch = 0
        smodel.train(False)
        
        while ex_epoch < options.num_epoch_pretrain:
            ex_epoch += 1
            #if ex_epoch % 5 == 0:
            #    print("Sub epoch ", epoch)
            
            batch_num = 0
            ex_epoch_loss = 0
            
            for sample_batched in dataloaderPre:
                batch_num = batch_num + 1
                input_batches = Variable(sample_batched['src']).transpose(0, 1)  # will give seq_len x batch_size
                input_lengths = sample_batched['src_len']
                
                output_batches = Variable(sample_batched['tgt']).transpose(0, 1)  # will give seq_len x batch_size
                output_lengths = sample_batched['tgt_len']
                control_batches = Variable(sample_batched['control_tensor'])  # this is  bs x control_len


                input_lengths, perm_idx = input_lengths.sort(0, descending=True)
                input_batches = input_batches[:, perm_idx]
                output_lengths = output_lengths[perm_idx]
                output_batches = output_batches[:, perm_idx]
                input_lengths = [x for x in input_lengths]
                output_lengths = [x for x in output_lengths]
                control_batches = control_batches[perm_idx, :]

                loss = train_batch(input_batches, input_lengths, output_batches, output_lengths, control_batches, encoder, decoder,
                            encoder_optimizer, options, vocab_tgt, smodel, scriterion)
                ex_epoch_loss+=loss
        print('Sub Epoch loss ', ex_epoch_loss / batch_num)
        
        if (epoch % options.generate_every == 0):
            generate(encoder, decoder, valid_lines, options, vocab_src, vocab_tgt, epoch, False, control_len)
        if(epoch%options.save_every == 0):
           save_model(encoder, decoder, vocab_src, vocab_tgt, encoder_optimizer, epoch, options)



def main():
    parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(parser)
    opts.train_opts(parser)
    opts.data_opts(parser)
    opts.score_opts(parser)
    options = parser.parse_args()

    print(options)

    argfile = options.save_model + '_arg.p'

    print('Saving arguments in ' + argfile)
    cPickle.dump(options, open(argfile, "wb"))

    train(options)

if __name__ == '__main__':
    main()

