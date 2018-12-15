import math, re
import kenlm
from collections import Counter

kenlm_path = "../ken_language_model/all.lm"
model = kenlm.Model(kenlm_path)
r = re.compile(r"(.+?)\1+")

def repetitions(s):
	global r
	s = s.replace("<EOS>","").replace(" ","")
	for match in r.finditer(s):
		yield (match.group(1), len(match.group(0))/len(match.group(1)))
		
def normalize_score(s, sent):
	invert = (-1/float(s))*10 #negative reciprocal log prob
	#print invert
	repeat = repetitions(sent)
	penalty = 1
	for r in repeat:
		#if more than one char and repeated more than once (not even "very very") 
		if len(r[0])>1 and r[1]>1:
			penalty+= r[1]
	freq = counts = Counter(sent.split())
	for w in freq.keys():
		if freq[w]>2:
			penalty+= freq[w]
	norm_score = invert / float(penalty)
	return norm_score

def kenlm_score(sent):
	sent = sent.replace("<EOS>","")
	if sent.strip()=="":
		return 0.0001
	exp_s = model.score(sent, bos = True, eos = True)
	score = normalize_score(exp_s, sent)
	return score

if __name__=="__main__":
	
	texts = ["i don't think this is a significant problem in practice. significant problem in practice. significant problem in practice.","scientists have the trouble explaining the evolution of competitive behavior",
	"the point site sorry is don't serve here",
	"the sorry point site blah ajbhbs",
	"there there there am am am am","<EOS>","seriously however dear dear dax i bed assembly bed dear dear assembly bed <EOS> <EOS>"]
	
	for s in texts:
		print(s+" : "+str(kenlm_score(s)))
