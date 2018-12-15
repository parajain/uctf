import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.rewards_helper import *
from modules.loss import *
from torch.nn import functional


def _cat_directions(h, bidirectional_encoder):
    if bidirectional_encoder:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h


def train_batch(input_batches, input_lengths, target_batches, target_lengths, control_batches, encoder, decoder, encoder_optimizer,
          options, vocab_tgt, smodel, scriterion, loss_function='masked-ce'):
    """
    :param input_batches: size max_seq_length x batch_size
    :param input_lengths:
    :param target_batches: size max_seq_length x batch_size
    :param target_lengths:
    :param control_batches: batch_size x control_len
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param options:
    :param vocab_tgt:
    :return:
    """

    encoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word
    batch_size = input_batches.size()[1]


    if options.use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        control_batches = control_batches.cuda()

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None, options)

    decoder_hidden = _cat_directions(encoder_hidden, encoder.bidirectional)

    decoder_input = Variable(torch.LongTensor([vocab_tgt.GO] * batch_size))
    #decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    if options.use_cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    for t in range(max_target_length):
        decoder_output, decoder_output_projected, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs, control_batches, options, False, True
        )
        all_decoder_outputs[t] = decoder_output_projected
        decoder_input = target_batches[t]

    if loss_function == 'binary-ce':
        loss_f = BCELossLogitsUsingTargets
    else:
        loss_f = masked_cross_entropy

    loss_ce = loss_f(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths, options
    )


    decoder_outputs = functional.softmax(all_decoder_outputs/options.temp, dim = 2)

    model_outputs = smodel(input_batches, decoder_outputs, False)
    control_batches = control_batches.long().squeeze(1)
    sloss = scriterion(model_outputs, control_batches - 1)

    loss = options.weight_ce * loss_ce + options.weight_mlp * sloss

    print('sloss ', sloss.data[0])
    print('loss_ce ', loss_ce.data[0])
    loss.backward()
    encoder_optimizer.step()
    return loss.data[0]

def train_batch_rl_tf(input_batches, input_lengths, control_batches, encoder, decoder, encoder_optimizer,
          options, vocab_tgt, avg_reward, scorer, epoch, use_vector,syn_f,sampled):
    """
    :param input_batches: size max_seq_length x batch_size
    :param input_lengths:
    :param target_batches: size max_seq_length x batch_size
    :param target_lengths:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param options:
    :param vocab_tgt:
    :return:
    """
    teacher_forcing = True
    encoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word
    batch_size = input_batches.size()[1]

    if options.use_cuda:
        input_batches = input_batches.cuda()
        control_batches = control_batches.cuda()

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None, options)

    decoder_input = Variable(torch.LongTensor([vocab_tgt.GO] * batch_size))

    decoder_hidden = _cat_directions(encoder_hidden, encoder.bidirectional)
    #decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(input_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    if options.use_cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    for t in range(max_target_length):
        decoder_output,decoder_output_projected , decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs, control_batches, options, False, True
        )
        all_decoder_outputs[t] = decoder_output_projected
        decoder_input = input_batches[t]  # Next input is current target

    
    new_src, new_tgt, curr_avg_reward, new_control = get_rewards_and_fake_targets_per_word(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        input_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        input_lengths, vocab_tgt, options, avg_reward, scorer,syn_f,sampled)

    """
    print(new_src)
    # test code
    new_control = []
    for i in range(len(new_src)):
        new_control.append([1] * 3)
    """

    return new_src, new_tgt,new_control,  curr_avg_reward


