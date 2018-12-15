import torch
import numpy as np
from torch.nn import functional
from torch.autograd import Variable
import nltk.translate.bleu_score as bs
from nltk.translate.bleu_score import SmoothingFunction
from masked_cross_entropy import  *

def create_rewards_extended_tensor(rewards, bs, msl, vs):
    """
    :param rewards: a numpy array of length batch size, representing reward per sequence
    :param bs: batch size
    :param msl: max seq length
    :param vs: vocab size or num of classes
    :return: return a matrix of size msl x bs x vs with elements replicated for msl and vocab dimension
    """
    rewards = np.tile(rewards, msl)
    rewards = np.reshape(rewards, (msl, bs))
    rewards = np.repeat(rewards, vs)
    rewards = np.reshape(rewards, (msl, bs, vs))
    return rewards

def create_rewards_extended_tensor_per_word(rewards, bs, msl, vs):

    rewards = np.repeat(rewards, vs)
    rewards = np.reshape(rewards, (msl, bs, vs))
    return rewards

def get_per_word_score(ref, sent, msl):
    rewards = np.zeros(msl)
    tlen = min(len(ref), len(sent))
    for idx in xrange(tlen):
        w = sent[idx]
        if w == ref[idx]:
            rewards[idx] = 1
        else:
            rewards[idx] = -1

    return rewards

def get_per_word_avg_reward(ref, sent, msl, avg_reward):
    rewards = np.zeros(msl)
    tlen = min(len(ref), len(sent))
    for idx in xrange(tlen):
        w = sent[idx]
        if w == ref[idx]:
            rewards[idx] = 2
        else:
            rewards[idx] = - 0.2
    cur_avg = rewards.sum()/ tlen
    if cur_avg < avg_reward:
        rewards[:tlen] = - 0.1
    return rewards

def get_per_word_reward_from_score(sent, msl, score,  avg_reward):
    rewards = np.zeros(msl) + score
    return rewards


def get_bleu_score(ref, sent):
    """
    :param ref: ordered list of tokens in input reference sentence
    :param sent: ordered list of tokens in generated sentence
    :return: bleu score
    refer: http://www.nltk.org/_modules/nltk/translate/bleu_score.html
    """
    #print(sent)
    #print(ref)
    bias = 0.6
    chencherry = SmoothingFunction()
    s = bs.sentence_bleu([ref], sent,weights=(0.25, 0.25, 0.25, 0.25)) + 0.0001
    #s = 0
    if s < bias:
        s = 0
    #s = s * 10000
    #print('###############################################BELU', s)
    #if s < 200:
    #    s = s * -1
    #else:
    #    s = s * 10
    return s



def get_rewards_and_fake_targets_per_word(logits, target, length, vocab_tgt, opts, avg_reward):
    """
    :param logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
    :param length:
    :param vocab_tgt: target vocab object
    :param opts: command line options
    :return: rewards tensor in shape max_len x batch x num_classes
            fake targets containing a LongTensor of size
            (batch, max_len)
    """
    batch_size = logits.data.shape[0]
    max_length = logits.data.shape[1]
    num_classes = logits.data.shape[2]
    fake_target_output = Variable(torch.zeros(batch_size, max_length).long())
    rewards = np.zeros([max_length, batch_size])
    sm_dump = open('probs.txt', 'a')
    if opts.use_cuda:
        fake_target_output = fake_target_output.cuda()
    for i in xrange(batch_size):
        decoder_output = logits[i]
        decoder_output = functional.softmax(decoder_output)
        decoded_words = []
        target_words = []
        actual_target_output = target[i]

        for di in range(max_length):
            probs = decoder_output[di]
            sm_dump.write(str(probs.data) + '\n')
            action = probs.multinomial().data
            topv, topi = decoder_output.data.topk(1)
            #prob = probs[action[0]]
            #ni = topi[0][0]
            ni = action[0]
            fake_target_output[i, di] = ni
            if ni == vocab_tgt.EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab_tgt.index2word(ni))

        for ti in range(length[i]):
            target_words.append(vocab_tgt.index2word(actual_target_output[ti].data[0]))

        score = get_per_word_avg_reward(target_words, decoded_words, max_length, avg_reward)
        rewards[:, i] = score

    #rewards = np.array(rewards)
    #avg_reward = rewards.sum() / len(rewards)
    #rewards = np.array([avg_reward] * len(rewards))
    curr_avg_reward = rewards.mean()
    expanded_rewards_tensor = create_rewards_extended_tensor_per_word(rewards, batch_size, max_length, num_classes)
    expanded_rewards_tensor = Variable(torch.from_numpy(expanded_rewards_tensor).float())
    sm_dump.close()
    return expanded_rewards_tensor, fake_target_output, curr_avg_reward









################################################

def get_cross_entropy_rewards_and_fake_targets(logits, target, length,loss_, vocab_tgt, opts):
    """
    :param logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
    :param length:
    :param vocab_tgt: target vocab object
    :param opts: command line options
    :return: rewards tensor in shape max_len x batch x num_classes
            fake targets containing a LongTensor of size
            (batch, max_len)
    """
    batch_size = logits.data.shape[0]
    max_length = logits.data.shape[1]
    num_classes = logits.data.shape[2]
    fake_target_output = Variable(torch.zeros(batch_size, max_length).long())
    rewards = []
    if opts.use_cuda:
        fake_target_output = fake_target_output.cuda()
    for i in xrange(batch_size):
        decoder_output = logits[i]
        decoder_output_sm = functional.softmax(decoder_output)
        decoder_output = decoder_output.view(1, max_length, num_classes)
        #print('&&&&&&&&&&&&&', decoder_output.size())
        decoded_words = []
        target_words = []
        #print('%%%%%%%%%%%%', target[i].size())
        #print(max_length)
        actual_target_output = target[i].view(1, target[i].size()[0])

        for di in range(max_length):
            probs = decoder_output_sm[di]
            #action = probs.multinomial().data
            topv, topi = decoder_output_sm.data.topk(1)
            #prob = probs[action[0]]
            ni = topi[0][0]
            #ni = action[0]
            fake_target_output[i, di] = ni
            if ni == vocab_tgt.EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab_tgt.index2word(ni))

        #   loss = masked_cross_entropy(decoder_output,  actual_target_output, length[i], opts)
        score = 1/loss_
        #print('Loss####### ', loss_)
        rewards.append(score.data.numpy())

    rewards = np.array(rewards)
    expanded_rewards_tensor = create_rewards_extended_tensor(rewards, batch_size, max_length, num_classes)
    expanded_rewards_tensor = Variable(torch.from_numpy(expanded_rewards_tensor).float())
    return expanded_rewards_tensor, fake_target_output


def get_rewards_and_fake_targets(logits, target, length, vocab_tgt, opts):
    """
    :param logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
    :param length:
    :param vocab_tgt: target vocab object
    :param opts: command line options
    :return: rewards tensor in shape max_len x batch x num_classes
            fake targets containing a LongTensor of size
            (batch, max_len)
    """
    batch_size = logits.data.shape[0]
    max_length = logits.data.shape[1]
    num_classes = logits.data.shape[2]
    fake_target_output = Variable(torch.zeros(batch_size, max_length).long())
    rewards = []
    if opts.use_cuda:
        fake_target_output = fake_target_output.cuda()
    for i in xrange(batch_size):
        decoder_output = logits[i]
        decoder_output = functional.softmax(decoder_output)
        decoded_words = []
        target_words = []
        actual_target_output = target[i]

        for di in range(max_length):
            probs = decoder_output[di]
            action = probs.multinomial().data
            topv, topi = decoder_output.data.topk(1)
            #prob = probs[action[0]]
            #ni = topi[0][0]
            ni = action[0]
            fake_target_output[i, di] = ni
            if ni == vocab_tgt.EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab_tgt.index2word(ni))

        for ti in range(length[i]):
            target_words.append(vocab_tgt.index2word(actual_target_output[ti].data[0]))

        score = get_bleu_score(target_words, decoded_words)
        rewards.append(score)

    rewards = np.array(rewards)
    #avg_reward = rewards.sum() / len(rewards)
    #rewards = np.array([avg_reward] * len(rewards))
    expanded_rewards_tensor = create_rewards_extended_tensor(rewards, batch_size, max_length, num_classes)
    expanded_rewards_tensor = Variable(torch.from_numpy(expanded_rewards_tensor).float())
    return expanded_rewards_tensor, fake_target_output



#############################################

def reward_loss(logits, target, length, vocab_tgt, opts):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    #target = target.cpu()
    batch_size = logits.data.shape[0]
    max_length = logits.data.shape[1]
    loss = Variable(torch.zeros(1, 1)) 
    print('*************Calculating reward loss******************************')
    gamma = 0.1
    
    TR = Variable(torch.zeros(1,1))
    R = Variable(torch.zeros(1,1))
    if opts.use_cuda:
        TR = TR.cuda()
        R = R.cuda()
    for i in xrange(batch_size):
        #print('Batch **** ', i)
        decoder_output = logits[i]
        #print('decoder_output shape')
        #print(decoder_output.data.shape)
        #print(max_length)
        decoder_output = functional.softmax(decoder_output)
        target_output = target[i]
        decoded_words= []
        target_words = []
        log_probs = []
        entropies = []
        l = 0
         
        seqR = Variable(torch.zeros(1,1))
        if opts.use_cuda:
            seqR = seqR.cuda()
        for di in range(max_length):
            #topv, topi = decoder_output[di].data.topk(1)
            #ni = topi[0]
            #log_probs.append(topv.log())
            #print(ni)
            probs = decoder_output[di]
            entropy = - (probs*probs.log()).sum()
            action = probs.multinomial().data
            prob = probs[action[0]]
            log_prob = prob.log()
            entropies.append(entropy)
            #log_probs.append(log_prob)
            log_probs.append(prob)
            ni = action[0]
            if ni == vocab_tgt.EOS:
                decoded_words.append('<EOS>')
                print('breaking')
                break
            else:
                decoded_words.append(vocab_tgt.index2word(ni))
        
        for ti in range(length[i]):
            target_words.append(vocab_tgt.index2word(target_output[ti].data[0]))
        score = get_bleu_score(target_words, decoded_words)
        R = score# + R * gamma
        #print('R')
        #print(R)
        lp = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        R = Variable( torch.from_numpy( np.float32(R).reshape((1,1)) ).cuda() ).expand_as(lp)
        #print('log pro')
        #print(lp)
        #print(R)
        #l = l + (lp * (Variable(R).expand_as(lp))).sum() #add bias
        #print('(lp * R).sum()')
        #print((lp * R).sum())
        #TR = TR - (lp * R).sum() #add bias, taking - lp * R because of loss
        seqR = seqR + ((lp * R ).sum()) - (0.01 * entropies).sum()

        TR = TR + seqR
        #print('****************seqR**************')
        #print(seqR)
        #print(TR)

    TR = TR/batch_size
    loss = -1 * TR
    print('loss')
    print(loss)
    return loss









