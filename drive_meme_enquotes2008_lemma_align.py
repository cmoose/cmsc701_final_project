# This code runs a word2vec alignment on memetracker raw text data from 2008
# Data had been preprocessed using CoreNLP via preprocess_cornell_quotes.py
#
# This code was never used for the report, as it's based on the memetracker raw text dataset (which we dropped)
# Author: Chris Musialek
# Date: Oct 2015
#

from gensim.models import Word2Vec
import os.path
import random
import run_global_alignment
import w2v_sub_matrix


def run_alignment():
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    cornell_en_quotes_lemma_file = 'data/en_quotes_2008-08.lemma.txt'
    w2v_bin_filename = 'data/en_quotes_2008-08.lemma.vectors.bin'
    en_quotes_data_fullpath = os.path.join(git_repo_path, cornell_en_quotes_lemma_file)

    #Raw scores of substitution matrix
    w2v_model = Word2Vec.load_word2vec_format(os.path.join(git_repo_path,w2v_bin_filename), binary=True)

    def load_data():
        print "Loading data ...{0}".format(en_quotes_data_fullpath)
        phrases = []
        fh = open(en_quotes_data_fullpath)
        for line in fh:
            phrases.append([x.strip() for x in line.strip().split()])
        print "Done loading..."
        return phrases

    #Create our w2v substitution matrix
    sub_matrix = w2v_sub_matrix.w2v_sub_matrix(w2v_bin_filename, 'en_quotes_2008-08.lemma')

    all_phrases = load_data()
    randint = random.randint(0,len(all_phrases))
    static_phrase = all_phrases[randint]

    #This is more of a test, runs a set of global alignments on one randomly chosen phrase
    pqs = run_global_alignment.run_global_alignments([static_phrase], all_phrases[0:1000], sub_matrix)

    run_global_alignment.print_priority_queues(pqs)

if __name__ == '__main__':
    run_alignment()