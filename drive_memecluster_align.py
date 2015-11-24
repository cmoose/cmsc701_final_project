# Drives the alignments of the memetracker phrase cluster data
#
# Author: Chris Musialek
# Date: Nov 2015

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_memetracker     #Loads the data for us
import word2vec
import os.path
from gensim.models import Word2Vec
import random


def load_data(raw_gz_fn, clusters_pkl_fn):
    return parse_memetracker.load_memetracker_data(raw_gz_fn, clusters_pkl_fn)


def load_data(w2v_phrases_fn):
    print "Loading data ...{0}".format(w2v_phrases_fn)
    fh = open(w2v_phrases_fn)
    all_phrases = []
    for line in fh:
        all_phrases.append([x.strip() for x in line.strip().split()])
    return all_phrases


# Loads the raw data, writes phrases to file, runs word2vec to create bin plus
# new phrases data (creates bigrams and trigrams smartly)
# w2v_basename example: 'memetracker-clusters-phrases'
def do_prep_work(w2v_basename):
    # Dwnld from http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz
    raw_gz_fn = 'data/clust-qt08080902w3mfq5.txt.gz'

    clusters_pkl_fn = 'pkl/memetracker-clusters.pkl'

    clusters = load_data(raw_gz_fn, clusters_pkl_fn)
    all_phrases = parse_memetracker.get_memtracker_phrases(clusters)
    parse_memetracker.write_phrases_to_file(all_phrases, os.path.join('data', w2v_basename + '.txt'))

    # Now, run word2vec
    word2vec.create_bin_file(w2v_basename)

    # We return the two filenames needed to run the alignments
    return w2v_basename + '.bin', w2v_basename + '-final'


def run_alignments():
    # Preprocess if needed
    w2v_basename = 'memetracker-clusters-phrases'
    if not (os.path.exists(os.path.join('data', w2v_basename + '.bin')) or
        (os.path.exists(os.path.join('data', w2v_basename + '-final')))):
        do_prep_work(w2v_basename)

    w2v_model = Word2Vec.load_word2vec_format(os.path.join('data', w2v_basename + '.bin'), binary=True)

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

    all_phrases = load_data(os.path.join('data', w2v_basename + '-final'))
    randint = random.randint(0,len(all_phrases))
    static_phrase = all_phrases[randint]

    pqs = run_global_alignment.run_global_alignments([static_phrase], all_phrases, word2vec_sub_matrix)

    run_global_alignment.print_priority_queues(pqs)


if __name__ == '__main__':
    run_alignments()