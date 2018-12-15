#####Important : Install NLTK with data
##### Install readability : https://pypi.python.org/pypi/readability


print("Importing modules for reward...")

from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import readability, codecs, os, string, re,pickle
import numpy as np

#print "Loading resources..."

base_path = os.path.dirname(os.path.abspath(__file__))
d = cmudict.dict()

"""
Load the dictionary containing the words in the Academic Word List (AWL).
"""
finput_awl = codecs.open(base_path+'/resources/awl.txt', 'r', 'utf-8')
awl_word_lines = finput_awl.readlines()
awl_words = map(lambda x: x.strip(), awl_word_lines)

finput_all = codecs.open(base_path+'/resources/all.txt', 'r', 'utf-8')
all_word_lines = finput_all.readlines()
all_words = map(lambda x: x.strip(), all_word_lines)

syn_resources = pickle.load(open(base_path+"/resources/syn.dat",'rb'))
print("Resources loaded...")

def normalize(text):
	text = text.lower()
	text = filter(lambda x: x in string.printable, text)
	for punct in string.punctuation:
		text = text.replace(punct," "+punct+" ")
	text = re.sub(' +',' ',text)
	return text
	
	
def nsyl(word):
	if word.lower() in d.keys():
		return sum([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
	else:
		return 0
		
def compute_avg_syllable_count(text):
	#syllable per word
	syllables_per_word = []
	syllables_per_word.extend(map(lambda x: nsyl(x), text.strip().split()))
	syl_avg_count = sum(syllables_per_word)/float(len(syllables_per_word))
	return syl_avg_count

def word_in_awl(word):
	global awl_words
	if word in string.punctuation:
		return 0
	lemma = wn.morphy(word)	
	if lemma is None:
		lemma = word
	if lemma in awl_words:
		return 1
	return 0
def is_formal(word):
	if word_in_awl(word)==1:
		return True
	else:
		return False
def valid_en_word(word):
	global all_words
	if word in string.punctuation:
		return False
	lemma = wn.morphy(word)
	if lemma is None:
		lemma = word
	if lemma in all_words:
		return True
	return False

def avg_formal_word_count(text):
	fw=0
	is_word_in_list = []
	is_word_in_list.extend(map(lambda x: word_in_awl(x), text.strip().split()))
	fw = sum(is_word_in_list) 
	avg_fw = fw / float(len(text.split()))
	return avg_fw

def compute_readability(text):
        score = float(len(s)) / float(len(s.split()))
        score = score / float(10)


def compute_textual_rewards(text):
	rewards = {}
	text = normalize(text)
	rewards['AVG_syllable'] = compute_avg_syllable_count(text)
	rewards['readability'] = compute_readability(text)
	rewards['formalwords'] = avg_formal_word_count (text)
	
	return rewards
def vect_reward_score(t,generated,max_length,nlp):
	#t = " ".join(t.replace("<EOS>","").replace("<OOV>","").strip().split())
	#generated = " ".join(generated.replace("<EOS>","").replace("<OOV>","").strip().split())
	vect = np.zeros(max_length)
	vvv = map(lambda x:(x.lemma_,x.tag_), nlp(t.decode("unicode-escape")))
	lemma_src, pos_src = zip(*vvv)
	
	lemma_tgt = map(lambda x:x.lemma_, nlp(generated.decode("unicode-escape")))
	min_len = min([len(lemma_src),len(lemma_tgt)])
	if min_len > max_length:
		#print "minlen exceeding..."
		min_len = max_length
	for i in range(min_len):
		src_wd = lemma_src[i]
		gen_wd = lemma_tgt[i]
		#print src_wd,gen_wd
		pos = pos_src[i]
		#print pos			
		if pos.lower()[0] == 'n':
			syn_list = syn_resources['n'].get(src_wd,[])
		elif pos.lower()[0] == 'v':
			syn_list = syn_resources['v'].get(src_wd,[])
		elif pos.lower()[0] == 'r':
			syn_list = syn_resources['r'].get(src_wd,[])
		elif pos.lower()[0]== 'j':
			syn_list = syn_resources['a'].get(src_wd,[])
			#print "yes"
		else:
			syn_list = []
		#print syn_list
		if syn_list==[]:
			if gen_wd==src_wd:
				vect[i] = 1.0
			else:
				vect[i] = -1.0
		else:
			if gen_wd in syn_list:
				vect[i] = 1.0
				#if is_formal(gen_wd):
				#	vect[i] = 1.0
				#else:
				#	vect[i] = 0.5
			else:
				vect[i] = -1.0
	return vect
def word_reward_score(w1,w2,nlp):
	score = 0.0
	vvv = map(lambda x:(x.lemma_,x.tag_), nlp(w1.decode("unicode-escape")))
	lemma_src, pos_src = zip(*vvv)

	lemma_tgt = map(lambda x:x.lemma_, nlp(w2.decode("unicode-escape")))
	src_wd = lemma_src[0]
	gen_wd = lemma_tgt[0]
	#print src_wd,gen_wd
	pos = pos_src[0]
	#print pos                      
	if pos.lower()[0] == 'n':
		syn_list = syn_resources['n'].get(src_wd,[])
	elif pos.lower()[0] == 'v':
		syn_list = syn_resources['v'].get(src_wd,[])
	elif pos.lower()[0] == 'r':
		syn_list = syn_resources['r'].get(src_wd,[])
	elif pos.lower()[0]== 'j':
		syn_list = syn_resources['a'].get(src_wd,[])
		#print "yes"
	else:
		syn_list = []
		#print syn_list
	if syn_list==[]:
		if gen_wd==src_wd:
			score = 1.0
		else:
			score = 0.0
	else:
		if gen_wd in syn_list:
			print ("Syn found: ",gen_wd)
			if is_formal(gen_wd):
				score = 1.0
			else:
				score = 0.1
		else:
			score = 0.0
	return score
def pretty_print(features):
	keys = features.keys()
	keys.sort()
	for k in keys:
		print(k+" : "+str(features[k]))
		
if __name__=="__main__":
	while (True):
		print("Enter a piece of text...")
		text = input()
		if text.strip()=="":
			exit(0)
		#rewards = compute_textual_rewards(text)
		#pretty_print(rewards)
		for word in text.split():
			print(word+" "+str(valid_en_word(word))+" "+str(is_formal(word)))
	
	
