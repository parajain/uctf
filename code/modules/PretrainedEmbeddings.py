import  numpy as np


def load_glove_embedding(filename, embed_size, id2word, word2id):
    """
    :param filename: golve embedding file, each line is like <word space embedding>
    :param embed_size:
    :param id2word:
    :param word2id:
    :return: numpy array of weights which can be used to initialize embeddings
    """
    #TODO:(Parag) we can remove the need of embed_size by peeking into the text file.
    # Not extracting embed_size to make sure we do not automatically load wrong embedding
    vocab_size = len(id2word)
    sd = 1 / np.sqrt(embed_size)  # Standard deviation to use
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)

    with open(filename, mode="r") as textFile:
        for line in textFile:
            # Separate the values from the word
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding weights
            id = word2id.get(word, None)
            if id is not None:
                weights[id] = np.array(line[1:], dtype=np.float32)
    return weights
