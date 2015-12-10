
import os.path
import cPickle
import math

blosum_scores = cPickle.load(open('./evaluation/blosum_submatrix.pkl','rb'))
wordlist = cPickle.load(open('./evaluation/blosum_wordlist.pkl'))

# Define our word2vec substitution matrix
def blosum_sub_matrix(x,y):
    model = blosum_scores

    if x not in wordlist or y not in wordlist:
        return 0


    if x > y:
        if model[y,x] != 0:
            return math.log(model[y, x],2)
    elif x < y:
        if model[x,y] != 0:
            return math.log(model[x,y],2)

    elif x == y:
        if model[x,y] != 0:
            return math.log(model[x,y],2)


    return 0



