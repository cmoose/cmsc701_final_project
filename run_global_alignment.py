# Code to run the global alignment using N-W algorithm.
# Code also parallelize's work across 8 cores.
#
# Author: Chris Musialek
# Date: Nov 2015
#

import numpy as np
from numpy import genfromtxt
from gensim.models import Word2Vec
import os.path
import random
import heapq
import multiprocessing
import pickle

#Globals
print_results = False


#Given two phrases and substitution matrix, run N-W algorithm to find optimal alignment
#Return optimal alignment and score
def single_global_align(X,Y, sub):

    def safe_get_char(X,index):
        if index == -1:
            return '-'
        else:
            return X[index]

    def backtrack(lastrow,lastcol):
        currow = lastrow
        curcol = lastcol
        curkey = T[currow,curcol]
        seq1 = []
        seq2 = []

        while currow >=0 and curcol >=0:
            if curkey == 0:
                seq1.append(safe_get_char(X,currow))
                seq2.append(safe_get_char(Y,curcol))
                currow -= 1
                curcol -= 1
            elif curkey == 1:
                seq1.append(safe_get_char(X,currow))
                seq2.append('-')
                currow -= 1
            elif curkey == 2:
                seq1.append('-')
                seq2.append(safe_get_char(Y,curcol))
                curcol -= 1
            curkey = T[currow,curcol]

        return seq1, seq2

    global_max_score = 0
    global_max_pos = [0,0]
    gap = 1
    S = np.empty((len(X), len(Y)), dtype=float) #score matrix
    for i in range(0,len(X)):
        S[i,0] = i*gap*-1
    for j in range(0,len(Y)):
        S[0,j] = j*gap*-1
    T = np.empty((len(X), len(Y)), dtype=int) #traceback
    T[0,0] = 0
    for i in range(1,len(X)):
        T[i,0] = 1 #always go up
    for j in range(1,len(Y)):
        T[0,j] = 2 #always go left
    for i in range(1,len(X)):
        for j in range(1,len(Y)):
            local_scores = []
            local_scores.append(S[i-1,j-1] + sub.get_score(X[i], Y[j])) #match/mismatch
            local_scores.append(S[i-1,j] - gap) #gap in Y
            local_scores.append(S[i,j-1] - gap) #gap in X
            local_max_score = max(local_scores)
            #store local max score in score matrix
            S[i,j] = local_max_score

            #find position of local max, this becomes our traceback key
            #0 = diag back
            #1 = upwards
            #2 = left
            traceback_key = local_scores.index(local_max_score)
            T[i,j] = traceback_key

            if S[i,j] > global_max_score:
                global_max_score = S[i,j]
                global_max_pos[0] = i
                global_max_pos[1] = j

    align_score = S[global_max_pos[0], global_max_pos[1]]
    seq1, seq2 = backtrack(len(X)-1, len(Y)-1)
    alignment = [seq1,seq2]

    return align_score, alignment


#Given two phrases and substitution matrix, run S-W algorithm to find
# optimal local alignment
# Return optimal alignment and score
# Author: Karthik Badam
def single_local_align(X,Y, sub):

    def safe_get_char(X,index):
        if index == -1:
            return '-'
        else:
            return X[index]

    def backtrack(lastrow,lastcol):
        currow = lastrow
        curcol = lastcol
        curkey = T[currow,curcol]
        seq1 = []
        seq2 = []

        while currow >0 and curcol >0:
            if curkey == 0:
                seq1.append(safe_get_char(X,currow-1))
                seq2.append(safe_get_char(Y,curcol-1))
                currow -= 1
                curcol -= 1
            elif curkey == 1:
                seq1.append(safe_get_char(X,currow-1))
                seq2.append('-')
                currow -= 1
            elif curkey == 2:
                seq1.append('-')
                seq2.append(safe_get_char(Y,curcol-1))
                curcol -= 1
            elif curkey == 3:
                if currow > 0:
                    seq1.append(safe_get_char(X,currow-1))
                    seq2.append('-')
                    currow -= 1

                if curcol > 0:
                    seq1.append('-')
                    seq2.append(safe_get_char(Y,curcol-1))
                    curcol -= 1

            curkey = T[currow,curcol]

        return seq1, seq2

    global_max_score = 0
    global_max_pos = [0,0]
    gap = 0
    S = np.empty((len(X)+1, len(Y)+1), dtype=float) #score matrix
    T = np.empty((len(X)+1, len(Y)+1), dtype=int) #traceback

    for i in range(0,len(X)+1):
        S[i,0] = 0
    for j in range(0,len(Y)+1):
        S[0,j] = 0

    T[0,0] = 0
    for i in range(1,len(X)+1):
        T[i,0] = 1 #always go up
        T[i,0] = 3
    for j in range(1,len(Y)+1):
        T[0,j] = 2 #always go left
        T[i,0] = 3
    for i in range(1,len(X)+1):
        for j in range(1,len(Y)+1):
            T[i,j] = 3
            local_scores = []
            local_scores.append(S[i-1,j-1] + sub.get_score(X[i-1], Y[j-1])) #match/mismatch
            local_scores.append(S[i-1,j] - gap) #gap in Y

            if Y[j-1] == "S-END":
                local_scores.append(S[i,j-1] - 100) #gap in X
            else:
                local_scores.append(S[i,j-1] - gap) #gap in X

            local_max_score = max(local_scores)
            S[i,j] = local_max_score

            #find position of local max, this becomes our traceback key
            #0 = diag back
            #1 = upwards
            #2 = left
            #3 = skip
            traceback_key = local_scores.index(local_max_score)

            if S[i, j] >= 0:
                T[i,j] = traceback_key
            else:
                S[i,j] = 0

            if S[i,j] > global_max_score:
                global_max_score = S[i,j]
                global_max_pos[0] = i
                global_max_pos[1] = j

    align_score = S[global_max_pos[0], global_max_pos[1]]
    seq1, seq2 = backtrack(global_max_pos[0], global_max_pos[1])
    alignment = [seq1,seq2]

    return align_score, alignment


# Nicely prints the alignment of two word phrases, using dashes for gaps
def print_alignment(alignment):
    seq1 = alignment[0]
    seq2 = alignment[1]
    seq1_formatted = []
    seq2_formatted = []
    for i in reversed(range(0,len(seq1))):
        tok_seq1 = seq1[i]
        tok_seq2 = seq2[i]
        longest_token_len = max(len(tok_seq1), len(tok_seq2))
        if tok_seq1.startswith("-"):
            seq1_formatted.append(tok_seq1.ljust(longest_token_len, "-"))
        else:
            seq1_formatted.append(tok_seq1.ljust(longest_token_len, " "))
        if tok_seq2.startswith("-"):
            seq2_formatted.append(tok_seq2.ljust(longest_token_len, "-"))
        else:
            seq2_formatted.append(tok_seq2.ljust(longest_token_len, " "))
    return " ".join(seq1_formatted) + "\n" + " ".join(seq2_formatted)


# This is the worker function for parallelization
# Encapsulates running 10,000 alignments [len(phrasesY) == 10000]
# against a single text
def multiple_global_align((phraseX, phrasesY, sub_matrix, chunk_region)):
    sub_pq = []
    print "Running global alignments for phrases {0}-{1}".format(chunk_region[0], chunk_region[1])
    for i, phraseY in enumerate(phrasesY):
        score, alignment = single_global_align(phraseX, phraseY, sub_matrix)
        heapq.heappush(sub_pq, (score, alignment))
    top_scores = heapq.nlargest(25, sub_pq)
    return top_scores


def eval_global_align(phraseX, phrasesY, sub_matrix):
    print "Running alignments between {0} and {1} phrases".format(phraseX, len(phrasesY))
    scores = [single_global_align(phraseX, phraseY, sub_matrix)[0] for phraseY in phrasesY]
    return scores


# This creates a datastructure of arguments for multiprocessing.Pool to run
# multiple_global_align in parallel
def build_pool_func_args(phraseX, phrasesY, sub_matrix):
    size = len(phrasesY)
    l = 0
    data = []
    # This will create chunks of 10,000 phrases in a format
    # suitable for using multiprocessing.Pool
    for r in range(1000,size,1000):
        func_args = []
        func_args.append(phraseX)       #First argument to multiple_global_align()
        func_args.append(phrasesY[l:r]) #Second argument to multiple_global_align()
        func_args.append(sub_matrix)    #Third argument to multiple_global_align()
        func_args.append([l,r])         #Fourth argument to multiple_global_align()
        l = r
        data.append(func_args)

    # Add last chunk of data
    func_args = []
    func_args.append(phraseX)
    func_args.append(phrasesY[l:size])
    func_args.append(sub_matrix)
    func_args.append([l,size])
    data.append(func_args)

    return tuple(data)


def run_global_alignments(phrasesX, phrasesY, sub_matrix):
    #For each phrase, run a global alignment on every other phrase
    #O(n^2)
    pqs = []
    #Parallelize!!
    #Use up to 8 cores at a time
    pool = multiprocessing.Pool(8)

    print "Running alignments..."
    for phraseX_id, phraseX in phrasesX.items():
        print "Aligning phrase '{0}' against {1} total phrases...".format(" ".join(phraseX), len(phrasesY))

        #This creates a datastructure of arguments for Pool to run
        #multiple_global_align in parallel
        data = build_pool_func_args(phraseX, phrasesY, sub_matrix)

        #Actually parallel process global alignments
        #Returns a list of the top 25 priority queues generated
        #from each chunk of phrases processed independently
        worker_results = pool.map(multiple_global_align, data)

        #We now need to find the topN of the items in all priority queues
        merged_scores = []
        for pq in worker_results:
            while pq:
                heapq.heappush(merged_scores, heapq.heappop(pq))
        top_scores = heapq.nlargest(25, merged_scores)

        pqs.append({'phraseX': phraseX, 'pq': top_scores})

        #Cache finished work to disk
        datasetname = sub_matrix.datasetname
        if not os.path.exists(os.path.join('pkl', datasetname)):
            os.mkdir(os.path.join('pkl', datasetname))
        pkl_fn = 'pkl/{0}/{1}.pkl'.format(datasetname, phraseX_id)
        pickle.dump({'phraseX': phraseX, 'pq': top_scores, 'id':phraseX_id}, open(pkl_fn, 'wb'))

    if print_results:
        #Print best alignments
        print_priority_queues(pqs)

    return pqs


# Print best alignments
def print_priority_queues(pqs):
    for obj in pqs:
        top_scores = obj['pq']
        phraseX = obj['phraseX']
        print "\nPrinting priority queue for: ", phraseX
    for score, alignment in top_scores:
        print score
        print print_alignment(alignment) + "\n"


#########################################
# Everything below used for testing only
#########################################
def simple_test():
    #Load and define substitution matrix
    LETTERS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    pam250 = genfromtxt('pam250.csv', delimiter=',')

    def sub_matrix(x,y):
        index_x = LETTERS.index(x)
        index_y = LETTERS.index(y)
        return int(pam250[index_x,index_y])

    #Run N-W algorithm
    score, alignment = single_global_align('MEANLYPRTEINSTRING', 'PLEASANTLYEINSTEIN', sub_matrix)
    print score
    print print_alignment(alignment)


def word2vec_test():
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    cornell_en_quotes_lemma_file = 'data/en_quotes_2008-08.lemma.txt'
    w2v_bin_filename = 'data/en_quotes_2008-08.lemma.vectors.bin'
    en_quotes_data_fullpath = os.path.join(git_repo_path, cornell_en_quotes_lemma_file)

    def load_data():
        print "Loading data ...{0}".format(en_quotes_data_fullpath)
        phrases = []
        fh = open(en_quotes_data_fullpath)
        for line in fh:
            phrases.append([x.strip() for x in line.strip().split()])
        print "Done loading..."
        return phrases

    #TODO: convert into bit-scores
    #Raw scores of substitution matrix
    w2v_model = Word2Vec.load_word2vec_format(os.path.join(git_repo_path,w2v_bin_filename), binary=True)

    #Define our word2vec substitution matrix
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

    #score, alignment = single_global_align(phraseX, phraseY, word2vec_sub_matrix)
    all_phrases = load_data()
    randint = random.randint(0,len(all_phrases))
    static_phrase = all_phrases[randint]

    #This is more of a test, runs a set of global alignments on one randomly chosen phrase
    run_global_alignments([static_phrase], all_phrases[0:1000], word2vec_sub_matrix)
