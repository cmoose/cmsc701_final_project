# Runs the global alignments of the memetracker phrase cluster data
# Dwnld raw data from http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz
#
# Author: Chris Musialek
# Date: Nov 2015
#

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_memetracker     #Loads the memetracker cluster dataset for us
import w2v_sub_matrix        #Contains the custom built word2vec substitution matrix
import train_word2vec        #Word2vec preprocessing driver
import os.path
import random
import re
import gzip


#Load the raw phrases (built from word2phrase iterations)
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
def do_prep_work_load(w2v_basename, raw_gz_fn):
    clusters_pkl_fn = 'pkl/{0}.pkl'.format(w2v_basename)

    # Load raw gz data, save as pkl file, return cluster datastructure
    clusters = parse_memetracker.load_memetracker_data(raw_gz_fn, clusters_pkl_fn)

    print "Getting all phrases from clusters..."
    all_phrases = parse_memetracker.get_memtracker_phrases(clusters)
    parse_memetracker.write_phrases_to_file(all_phrases, os.path.join('data', w2v_basename + '.txt'))


# Utility function used to retrieve number of completed processed alignments.
def get_completed_phrases():
    l = os.listdir('./pkl')
    regex = re.compile('^([0-9].*)\.pkl') #all filenames of all digits plus .pkl

    compl = [int(regex.search(x).group(1)) for x in l if regex.search(x)]

    return compl


# Actually run N-W (in parallel) on the memetracker cluster data
def run_alignments(w2v_bin_fn):
    # Raw data
    # Dwnld from http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz
    raw_gz_fn = 'data/clust-qt08080902w3mfq5.txt.gz'

    dataset_basename = 'memetracker-clusters-phrases'
    memetracker_phrases_fn = 'data/{0}-final'.format(dataset_basename)

    # Preprocess if needed including loading data and training the model
    if not (os.path.exists(w2v_bin_fn) or os.path.exists(os.path.join(memetracker_phrases_fn))):
        do_prep_work_load(dataset_basename, raw_gz_fn)
        train_word2vec.create_bin_file(dataset_basename)

    # word2phrase creates bigrams/trigrams, retokenizing original data, so we load this data instead
    all_phrases = load_data(memetracker_phrases_fn)

    #randints = random.sample(range(0,len(all_phrases)), 1) #Sample a random phrase from the full list
    randints = [131959]  #'what does not kill us makes us stronger'

    static_phrases = {}
    for i in randints:
        static_phrases[i] = all_phrases[i]

    #Create substitution matrix object
    sub_matrix = w2v_sub_matrix.w2v_sub_matrix(w2v_bin_fn, dataset_basename)

    # All the hard work is done here. Align a subset of phrases against the entire set
    # O(n^2) - N-W global align algorithm is parallelized to use 8 cores concurrently
    pqs = run_global_alignment.run_global_alignments(static_phrases, all_phrases, sub_matrix)

    # Print the results
    run_global_alignment.print_priority_queues(pqs)


if __name__ == '__main__':
    # Word2vec binary filename we'll use for the substitution matrix
    w2v_bin_fn = 'data/memetracker-clusters-phrases.bin' #Use this if you want to train a word2vec vector space
    #w2v_bin_fn = 'data/GoogleNews-vectors-negative300.bin' #Use this if you want to use pre-trained w2v vector space

    run_alignments(w2v_bin_fn)