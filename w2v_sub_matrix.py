# Definition of custom built word2vec substitution matrix using word2vec binary files
# trained on some textual data
#
# Author: Chris Musialek
# Date: Nov 2015
#

from gensim.models import Word2Vec
import os.path

w2v_basename = 'memetracker-clusters-phrases'
w2v_model = Word2Vec.load_word2vec_format(os.path.join('data', w2v_basename + '.bin'), binary=True)


# Define our word2vec substitution matrix
def word2vec_sub_matrix(x,y):
    model = w2v_model
    try:
        S_ij = model.similarity(x,y)
    except KeyError:
        if x == y:
            S_ij = 1
        else:
            S_ij = -1
    return float(S_ij)

