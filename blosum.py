# Definition of custom built word2vec substitution matrix using word2vec binary files
# trained on some textual data
#
# Author: Chris Musialek
# Date: Nov 2015
#


import os.path
import cPickle
import math

blosum_scores = cPickle.load(open('./evaluation/blosum_submatrix.pkl','rb'))
wordlist = cPickle.load(open('./evaluation/blosum_wordlist.pkl'))

# Define our word2vec substitution matrix
def blosum_sub_matrix(x,y):
    model = w2v_model

    if x not in wordlist or y not in wordlist:
        return 0

    if x > y:
        return math.log(model[y, x],2)
    elif x < y:
        return math.log(model[x,y],2)

    elif x == y:
        return math.log(model[x,y],2)


