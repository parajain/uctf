import torch
import numpy as np
from torch.nn import functional
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction
from modules.masked_cross_entropy import *
import hashlib
import json
from torch import topk as topK

np.random.seed(1)


def encode_hash(str1):
    str1 = str1.strip().replace("EOS", "")
    newstr = "".join(str1.split())
    hash1 = str(hashlib.md5(str1).hexdigest())
    return hash1


def create_rewards_extended_tensor_per_word(rewards, bs, msl, vs):
    rewards = np.repeat(rewards, vs)
    rewards = np.reshape(rewards, (msl, bs, vs))
    return rewards


def get_per_word_reward_from_score(msl, score):
    rewards = np.zeros(msl) + score
    return rewards


def get_per_word_reward_from_score_modified(msl, scoreV, score):
    reward1 = np.zeros(msl) + score
    reward2 = scoreV
    rewards = 0.3 * reward1 + 0.7 * reward2
    return rewards


def get_per_word_reward_controlled(msl, ctrl):
    rewards = np.zeros(msl) + ctrl
    return rewards


def get_per_word_same_output_score(ref, sent, msl):
    rewards = np.zeros(msl)
    tlen = min(len(ref), len(sent))
    for idx in xrange(tlen):
        w = sent[idx]
        if w == ref[idx]:
            rewards[idx] = 1
        else:
            rewards[idx] = -0.2
    return rewards


def get_per_word_setoverlap_score(ref, sent, msl):
    rewards = np.zeros(msl)
    tlen = min(len(ref), len(sent))
    for idx in xrange(tlen):
        w = sent[idx]
        if w in ref:
            rewards[idx] = 0.1
        else:
            rewards[idx] = - 0.1
    return rewards


def length_score(target_words, generated_words):
    tlen = len(target_words)
    glen = len(generated_words)
    d = tlen - glen
    if d > 0.3 * tlen:
        return - 0.1
    else:
        return 1


def check_avg(avg_reward, rewards, baseline):
    cur_avg = rewards.sum() / len(rewards)
    if cur_avg < avg_reward or cur_avg < baseline:
        rewards[:] = 0
    return rewards


def sample_single(decoder_output, max_length, vocab_tgt):
    decoded_words = []
    for di in range(max_length):
        probs = decoder_output[di]
        # action = probs.multinomial().data
        # ni = action[0]
        topv, topi = topK(probs, 1)  # probs.data.topk(1)
        # topv, topi = probs.data.topk(1)
        #ni = topi[0].data[0]
        ni = topi[0].item()
        # print ni
        if ni == vocab_tgt.EOS:
            decoded_words.append('<EOS>')
            # break
        else:
            decoded_words.append(vocab_tgt.index2word(ni))
    g = ' '.join(decoded_words)
    # print g
    return g


def sample_sentence(decoder_output, max_length, vocab_tgt, syn_func):
    decoded_words = []
    syn_flag = 0
    for di in range(max_length):
        probs = decoder_output[di]
        # action = probs.multinomial().data
        # ni = action[0]
        topv, topi = topK(probs, 1)  # probs.data.topk(1)
        ni = topi[0].data[0]
        # print ni
        if ni == vocab_tgt.EOS:
            decoded_words.append('<EOS>')
            break
        else:
            wr = vocab_tgt.index2word(ni.item())
            syns = []
            syns1 = syn_func['n'].get(wr, [])
            syns2 = syn_func['v'].get(wr, [])
            syns3 = syn_func['r'].get(wr, [])
            syns4 = syn_func['a'].get(wr, [])
            syns = syns1 + syns2 + syns3 + syns4
            if syns != [] and len(wr) > 3:
                syns.append(wr)
                syn_flag = 1
                # syns.sort(key=len,reverse=True)
                ws = np.random.choice(syns)
                ni = vocab_tgt.word2index(ws)
                decoded_words.append(ws)
            else:
                decoded_words.append(vocab_tgt.index2word(ni.item()))
    g = ' '.join(decoded_words)
    return g, syn_flag


def sample_sentences(decoder_output, max_length, vocab_tgt, syn_func, sampled, n):
    sentences = []
    max_try = 2 * n
    for i in xrange(max_try):
        if len(sentences) > n - 1:
            break
        # sentence = sample_sentence(decoder_output, max_length, vocab_tgt,syn_func)
        for try1 in range(200):
            # print "Trying: "+str(try1)
            sentence, syn_flag = sample_sentence(decoder_output, max_length, vocab_tgt, syn_func)
            if syn_flag == 0:
                # sentences.append(sentence)
                break
            hash1 = encode_hash(sentence)
            if hash1 not in sampled:
                sampled.add(hash1)
                # sentences.append(sentence)
                break
        if sentence not in sentences:
            sentences.append(sentence)
    # sentences.append("aaaaaaa") #just incase nothing is sampled
    return sentences


def get_cumulative_rewards(sentence, t, scorer, max_length):
    tt = t.replace("<EOS>", "").split()
    generated = sentence.replace("<EOS>", "").split()
    tt = " ".join(tt)
    generated = " ".join(generated)
    if tt == generated:
        return [-500, np.zeros(max_length)]
    try:
        scoreV = scorer.vect_score(tt, generated, max_length)
        score = np.sum(scoreV)
    except:
        scoreV = np.zeros(max_length)
        score = -500
    return [score, scoreV]


def all_rewards(t, generated, scorer, opts):
    #class_score = scorer.evaluate_ploarity(generated)
    # doc_sim_reward = scorer.doc_sim(generated, t)
    # score_overlap = get_per_word_same_output_score(target_words, decoded_words, max_length)
    try:
        ss_score = scorer.doc_sim(generated, t)  # scorer.short_sen_sim(t, generated)
    except:
        ss_score = 0.0
    try:
        lm_score = scorer.lang_model_score(generated)
    except:
        lm_score = 0.0
    try:
        formal_wc = scorer.avg_formal_word_count(generated)
    except:
        formal_wc = 0.0
    try:
        max_length = opts.src_seq_length
        readability = scorer.readability(t, max_length)
    except:
        readability = 0.0
    target_words = t.split()
    decoded_words = generated.split()
    len_score = length_score(target_words, decoded_words)
    #total_score = (readability * opts.fw_w + lm_score * opts.lm_w + opts.c_w * class_score + opts.dc_w * ss_score)
    total_score = (readability * opts.fw_w + lm_score * opts.lm_w  + opts.dc_w * ss_score)
    total_score = (total_score * (1 - opts.len_w) + len_score * opts.len_w) / 2

    # reward_package = [total_score, class_score, ss_score, lm_score, readability, len_score]
    reward_package = [total_score, ss_score, lm_score, readability]
    return reward_package


def get_rewards_and_fake_targets_per_word(logits, input, length, vocab_tgt, opts, avg_reward, scorer, syn_func,
                                          sampled):
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
    new_src = []
    new_tgt = []
    reward_list = []
    batch_size = logits.data.shape[0]
    max_length = logits.data.shape[1]
    num_classes = logits.data.shape[2]
    eps = 1 / max_length
    rewards = np.zeros([max_length, batch_size])
    # sm_dump = open('probs.txt', 'a')
    debug_file = open(opts.save_model + "_debug.txt", 'a')
    for i in xrange(batch_size):
        decoder_output = logits[i]
        decoder_output = functional.softmax(decoder_output)
        decoded_words = []
        target_words = []
        actual_target_output = input[i]
    
        for ti in range(length[i]):
            target_words.append(vocab_tgt.index2word(actual_target_output[ti].item()))
        t = ' '.join(target_words)

        generated = None
        reward_package = None

        generated_default = None
        generated_sampled = None
        scoreV_sampled = None
        scoreV_default = None

        ###first get sentence as per normal softmax
        sentence = sample_single(decoder_output, max_length, vocab_tgt)
        generated_default = sentence
        # print ('Default',generated_default)
        cum_rew_default, scoreV_default = get_cumulative_rewards(generated_default, t, scorer, max_length)
        reward_package_default = all_rewards(t, generated_default, scorer, opts)

        ### Now sample to see if anything better comes up
        mx_reward = -1000
        # print "Sampling...."
        while mx_reward == -1000:
            sentences = sample_sentences(decoder_output, max_length, vocab_tgt, syn_func, sampled, 5)
            for idx, sentence in enumerate(sentences):
                # generated = get sentence with best reward
                cumulative_score, scoreV_temp = get_cumulative_rewards(sentence, t, scorer, max_length)
                # print ("Cumulative Score",cumulative_score)
                if cumulative_score > mx_reward:
                    mx_reward = cumulative_score
                    generated_sampled = sentence
                    cum_rew_sampled = cumulative_score
                    scoreV_sampled = scoreV_temp
        reward_package_sampled = all_rewards(t, generated_sampled, scorer, opts)
        # print "Xampled..."
        # if (reward_package_sampled[0] >  reward_package_default[0]) and (reward_package_sampled[3] >  (reward_package_default[3]/2.0)) and (cum_rew_sampled > cum_rew_default):
        # if (reward_package_sampled[0] >  reward_package_default[0]) and (reward_package_sampled[4] >  reward_package_default[4]) and (cum_rew_sampled > cum_rew_default):
        if (reward_package_sampled[0] > reward_package_default[0]) and (cum_rew_sampled > cum_rew_default):
            # Only consider sampled output if it does better than default
            generated = generated_sampled
            scoreV = scoreV_sampled
            # score =  get_per_word_reward_from_score_modified(max_length,scoreV,total_score) #uncomment for variable reward
            reward_package = reward_package_sampled
            if opts.debug:
                print (" Sampled Sentence", generated)
        else:
            generated = generated_default
            reward_package = reward_package_default
            scoreV = scoreV_default
            if opts.debug:
                print ("Default Sentence", generated)
                print ("No need to append")
            continue
        t = t.replace("<EOS>", "").strip()
        generated = generated.replace("<EOS>", "").strip()
        new_src.append(t)
        generated = scorer.remove_redundant(generated)


        # logic for rewards computation
        reward_formal_t = reward_package[4]
        reward_formal_s = scorer.readability(t, max_length)
        reward_formal_ratio = reward_formal_t / float(reward_formal_s)

        current_control = 0

        #if opts.debug:
        #    print ("reward_formal_ratio", reward_formal_ratio)
        if reward_formal_ratio <= 1:
            current_control = 1
            reward_list.append([1])
        elif reward_formal_ratio > 1.005 and reward_formal_ratio <= 1.01:
            reward_list.append([2])
            current_control = 2
        elif reward_formal_ratio > 1.01 and reward_formal_ratio <= 1.02:
            reward_list.append([3])
            current_control = 3
        else: #reward_formal_ratio > 1.02:
            reward_list.append([4])
            current_control = 4
        #else:
        #    reward_list.append([1])
        #    current_control = 1

        if current_control > 1:
            new_tgt.append(generated)
            selected_sen = generated
        else:
            new_tgt.append(t)
            selected_sen = t


        debug_json = {}
        if opts.debug:
            print('DEBUG statement ' + '\n')
            debug_json['Input:: '] = t
            debug_json['Generated:: '] = generated
            debug_json['selected_sen:: '] = selected_sen
            debug_json['current_control:: '] = str(current_control)
            debug_json['Scores:: '] = str(reward_package)
            debug_json['reward_formal_ratio:: '] = str(reward_formal_ratio)
            d = json.dumps(debug_json, indent=4, sort_keys=True)
            print(d)
        debug_file.write(json.dumps(debug_json) + '\n')
            #print ('total_score, class_score, ss_score, lm_score, readability', str(reward_package))
        score = get_per_word_reward_from_score_modified(max_length, scoreV, reward_package[0])
        score_ = score  # check_avg(avg_reward, score, opts.baseline)
        rewards[:, i] = score_

    curr_avg_reward = rewards.mean()
    print('batch average', curr_avg_reward)

    # sample by repeitition
    # for i in range(3):
    #    new_src = new_src + new_src
    #    new_tgt = new_tgt + new_tgt
    #    reward_list = reward_list + reward_list
    return new_src, new_tgt, curr_avg_reward, reward_list
