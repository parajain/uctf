from __future__ import unicode_literals

import argparse
import spacy, string,sys
import readability # version 0.2
#from word_language_model.LMScore import *
#from modules.short_sentence_similarity import *
from textual_rewards.reward_polite import *
#from eval_forminf.evaluator_forminf import *
from kenlm_scorer import kenlm_score
#from levensthein import lev_dist
import numpy as np
class Scorer(object):
    def __init__(self, options=None):

        #parserCNN = argparse.ArgumentParser(description='Evalaute-CNN text Polarity Classifier')
        #parserCNN.add_argument('-snapshot_forminf', type=str,
        #                    default='eval_forminf/snapshot_forminf/snapshot_steps61000.pt',
        #                    help='filename of model snapshot')
        #parserCNN.add_argument('-srcfile_polarity', default="eval_forminf/data/train_src_forminf.txt",
        #                    help='source file path for vocab')

        #self.optionsCNN, self.argsCNN = parserCNN.parse_known_args()
        #print(self.argsCNN)

        #self.eval_obj = Evaluator_Forminf(self.optionsCNN)


        self.nlp = spacy.load('en_core_web_sm')
        #self.lm_snapshot = options.lm_snapshot
        #self.lmscorer = LMScore(self.lm_snapshot)


    #def evaluate_ploarity(self, s):
    #    score_positive = self.eval_obj.evalautor_score_positive(s, self.optionsCNN)
    #    return score_positive


    def doc_sim(self, s1, s2):
        #s1, s2 = tup
        s1 = s1.replace("<EOS>","").strip()
        s2 = s2.replace("<EOS>","").strip()
        #ddist = lev_dist(s1.split(),s2.split())
        if s1==s2:
            return -1
        s1 = set(s1.split())
        s2 = set(s2.split())
        diff1 = " ".join(list(s1-s2)).strip()
        diff2 = " ".join(list(s2-s1)).strip()
        if diff1=="" or diff2=="":
            return -1
        doc1 = self.nlp(diff1.decode('unicode-escape'))
        doc2 = self.nlp(diff2.decode('unicode-escape'))
        sim = doc1.similarity(doc2)
        #sim = sim /float(ddist) #penalizing
        return sim

    def compute_readability(text,length_normalize=False):
        #readability.getmeasures(text)
        R = readability.getmeasures(text.decode('unicode-escape'), lang=u'en', merge=True)
        score = R["Kincaid"] #Flesh - Kincaid score
        if length_normalize:
            score = score / float(len(text.split()))
        return score

    def lang_model_score(self, s):
        score = kenlm_score(s)
        #score = self.lmscorer.get_score(s)
        #print('Score :: ', score.data[0])
        return score

    #def avg_formal_word_count(self, s):
    #    wc = avg_formal_word_count(s)
    #    return wc
    
    def readability(self,s,maxlength):
        score = float(len(s)) / float(len(s.split()))
        score = score / float(10)
        #score = compute_readability(s)
        #score = score / float(len(s.split()))
        return score
    
    #def vect_score(self,t,generated,max_length):
    #    score = vect_reward_score(t,generated,max_length,self.nlp)
    #    return score
    
    #def word_wise_score(self,w1,w2):
    #   score = word_reward_score(w1,w2,self.nlp)
    #    return score
    
    def remove_redundant(self,text):
        text = text.replace("<EOS>","")
        newtext = ""
        prev = ""
        for t in text.split():
            if t!=prev:
                newtext +=t+" "
            prev = t
        return newtext.strip()
