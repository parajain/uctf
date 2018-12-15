from gensim.models import Word2Vec,KeyedVectors
import logging, numpy, cPickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def pre_compute_similarity_matrix(vocab):
	model = KeyedVectors.load_word2vec_format("glove100.dat", binary=False)
	most_simialr_indices = {}
	for w in vocab.keys():
		try:
			sim_indices = []
			index = vocab[w]
			sim_indices.append(index)
			similar_words = model.most_similar(w)
			for simw in similar_words:
				idx = vocab.get(simw, -1)
				if idx != -1:
					sim_indices.append(idx)
			most_simialr_indices[w] = sim_indices
		except:
			most_simialr_indices[w] = [vocab[w]]
	return most_simialr_indices

def vector_builder_for_ce(text, simmat):
	total_indices = []
	vocab_len = len(simmat.keys())
	vinit = numpy.zeros(vocab_len)
	words = set(text.split())
	for w in words:
		indices = simmat.get(w,[])
		if indices!=[]:
			total_indices+=indices
	total_indices = list(set(total_indices))
	
	prob = 1 / float(len(total_indices))
	
	for i in total_indices:
		vinit[i] = prob
	return vinit
	
def vectorize(text):
	with open("simmat.dat","rb") as f:
		simmat = cPickle.load(f)
		return vector_builder_for_ce(text, simmat)
		
if __name__=="__main__":
	
	#First prepare the simmat file if not done already
	dummy_vocab = {'i':1,'me':2,'myself':3,'you':4,'he':5,'formal':6,'informal':7}
	dummy_text = "i formal"
	
	print 'Preparing simmat file'
	simmat = pre_compute_similarity_matrix(dummy_vocab)
	cPickle.dump(simmat,open("simmat.dat","wb"))
	print vectorize(dummy_text)	
