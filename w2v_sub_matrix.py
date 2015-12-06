# Module to define the custom built word2vec substitution matrix using word2vec binary files
# trained on some textual data. It defaults to the custom built memetracker clusters w2v model built with word2vec.py,
# but can be replaced with other pre-trained w2v models as well, such as Google News, downloadable from:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#
# Note: This file must include the model on its own for multiprocessing.Pool to serialize properly. This allows the
# Needleman-Wunsch algorithm to run in parallel.
#
# Author: Chris Musialek
# Date: Nov 2015
#

from gensim.models import Word2Vec

google_w2v_bin_fn = 'data/GoogleNews-vectors-negative300.bin'
custom_w2v_bin_fn = 'data/memetracker-clusters-phrases.bin'

# Load the model from the data/ directory
w2v_model = Word2Vec.load_word2vec_format(custom_w2v_bin_fn, binary=True)

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

