# Class to define the custom built word2vec substitution matrix using word2vec binary files
# trained on some textual data. It defaults to the custom built memetracker clusters w2v model built with train_word2vec.py,
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

class w2v_sub_matrix:
    def __init__(self, custom_w2v_bin_fn, datasetname):
        # Load the model from the data/ directory
        print "Loading {0} w2v model...".format(custom_w2v_bin_fn)
        self.w2v_model = Word2Vec.load_word2vec_format(custom_w2v_bin_fn, binary=True)
        self.datasetname = datasetname

    # Define our word2vec substitution matrix
    def get_score(self, x, y):
        try:
            S_ij = self.w2v_model.similarity(x,y)
        except KeyError:
            #Neither tokens/words is found in word2vec model
            if x == y:
                S_ij = 1  #Default score of +1 if words match
            else:
                S_ij = -1
        return float(S_ij) #Default score of -1 if words do not match

