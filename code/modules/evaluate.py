import torch
import numpy as np
from torch.nn import functional
from torch.autograd import Variable
import nltk.translate.bleu_score as bs


def _cat_directions(h, bidirectional_encoder):
    if bidirectional_encoder:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

def evaluate(encoder, decoder, src_sentence, max_length, vocab_src, vocab_tgt, options, pretraining, control_len):
    print('evaluate ********************************')
    input_seqs, input_length = vocab_src.sentence_to_word_ids(src_sentence, max_length)
    input_lengths = [input_length]
    input_seqs = torch.LongTensor(input_seqs).unsqueeze(1)
    input_batches = Variable(input_seqs, volatile=True)

    control_list = [1] * control_len

    def get_control(s):
        # slist = s.split(' ')
        control_list = [float(x) for x in s]
        control_list = np.asarray(control_list, dtype='float32')
        control_tensor = torch.from_numpy(control_list)
        return control_tensor

    control_batches = get_control(control_list)
    control_batches = control_batches.unsqueeze(0)
    control_batches = Variable(control_batches)

    if options.use_cuda:
        input_batches = input_batches.cuda()
        control_batches = control_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)




    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None, options)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([vocab_tgt.GO]), volatile=True)  # SOS
    #decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoder_hidden = _cat_directions(encoder_hidden, encoder.bidirectional)

    if options.use_cuda:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    if options.decoder != 'rnn':
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        if options.pretrain_decoder or (not pretraining):
            decoder_output, decoder_output_projected, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, control_batches, options, False, True
            )
        else:
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, options
            )
            decoder_output_projected = decoder_output

        if options.decoder != 'rnn':
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        decoder_output_projected = functional.softmax(decoder_output_projected)
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
        #decoder_input = decoder_output_projected

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    if options.decoder != 'rnn':
        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]
    else:
        return decoded_words, None

def get_bleu_score(ref, sent):
    """
    :param ref: ordered list of tokens in input reference sentence
    :param sent: ordered list of tokens in generated sentence
    :return: bleu score
    refer: http://www.nltk.org/_modules/nltk/translate/bleu_score.html
    """
    try:
        s = bs.sentence_bleu([ref], sent)
    except ZeroDivisionError:
        return 0
    return s



def generate(encoder, decoder, input_sentences, options, vocab_src, vocab_tgt, epoch, pretraining, control_len):
    max_length = options.src_seq_length
    output_sentences = []
    num_sentences = len(input_sentences)
    total_b = 0
    for idx in xrange(num_sentences):
        input_sentence = input_sentences[idx]
        output_words, attentions = evaluate(encoder, decoder, input_sentence, max_length, vocab_src, vocab_tgt,
                                            options, pretraining, control_len)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
        input_sentences_words = input_sentence.split(' ')[:max_length]
        input_sentence = ' '.join(input_sentences_words)
        print('Input:::', input_sentence)
        print('Output::: ', output_sentence)
        bs = get_bleu_score(input_sentence, output_sentence)
        total_b = total_b + bs
    print(total_b)

################
def evaluate_all(encoder, decoder, input_sentences, ref_sentences, opts, vocab_src, vocab_tgt, epoch):
    max_length = opts.src_seq_length
    output_sentences = []
    total_bleu = 0
    num_sentences = len(input_sentences)
    for idx, input_sentence in enumerate(input_sentences):
        output_words, attentions = evaluate(encoder, decoder, input_sentence, max_length, vocab_src, vocab_tgt, opts)
        ref = ref_sentences[idx]
        bs = get_bleu_score(ref.split(' '), output_words)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
        total_bleu = total_bleu + bs
        #print('Input:::', input_sentence)
        #if ref is not None:
        #    print('reference::: ', ref)
        #print('Output::: ', output_sentence)
    print('-------------------> BLEU VALID:: ', total_bleu / num_sentences)
    with open('bleu_ce.txt', 'a') as f:
        f.write(str(total_bleu / num_sentences) + '\n')

    #with open(opts.save_model + '_generated_' + str(epoch), 'w') as ofile:
    #    ofile.write('-------------------> BLEU VALID:: '+ ' ' + str(total_bleu / num_sentences) + '\n')
    #   ofile.writelines(output_sentences)