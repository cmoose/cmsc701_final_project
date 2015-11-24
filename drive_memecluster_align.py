# Drives the alignments of the memetracker phrase cluster data
#
# Author: Chris Musialek
# Date: Nov 2015
#

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_memetracker     #Loads the memetracker cluster dataset for us
import w2v_sub_matrix        #Contains the custom built word2vec substitution matrix
import word2vec              #Word2vec preprocessing driver
import os.path
import random


# Load raw gz data, save as pkl file, return cluster datastructure
def load_data(raw_gz_fn, clusters_pkl_fn):
    return parse_memetracker.load_memetracker_data(raw_gz_fn, clusters_pkl_fn)


# Load the raw phrases (built from word2phrase iterations)
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
def do_prep_work(w2v_basename, raw_gz_fn):

    clusters_pkl_fn = 'pkl/{0}.pkl'.format(w2v_basename)

    clusters = load_data(raw_gz_fn, clusters_pkl_fn)
    all_phrases = parse_memetracker.get_memtracker_phrases(clusters)
    parse_memetracker.write_phrases_to_file(all_phrases, os.path.join('data', w2v_basename + '.txt'))

    # Now, run word2vec
    word2vec.create_bin_file(w2v_basename)

    # We return the two filenames needed to run the alignments
    return w2v_basename + '.bin', w2v_basename + '-final'


def run_alignments():
    # Raw data
    # Dwnld from http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz
    raw_gz_fn = 'data/clust-qt08080902w3mfq5.txt.gz'

    # Preprocess if needed
    w2v_basename = 'memetracker-clusters-phrases' #basename we'll use for all remaining files created
    if not (os.path.exists(os.path.join('data', w2v_basename + '.bin')) or
        (os.path.exists(os.path.join('data', w2v_basename + '-final')))):
        do_prep_work(w2v_basename, raw_gz_fn)

    #word2phrase creates bigrams/trigrams (new tokens in phrases), so we load this data instead
    all_phrases = load_data(os.path.join('data', w2v_basename + '-final'))

    #TODO: pick 1000 phrases at random
    randint = random.randint(0,len(all_phrases))
    static_phrase = all_phrases[randint]

    pqs = run_global_alignment.run_global_alignments([static_phrase], all_phrases, w2v_sub_matrix.word2vec_sub_matrix)

    run_global_alignment.print_priority_queues(pqs)


if __name__ == '__main__':
    run_alignments()