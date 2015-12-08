# Drives the alignments of the microsoft phrase cluster data.
# Outputs the top alignments for each text
#
# Author: Chris Musialek
# Date: Nov 2015
#

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_microsoft       #Loads the microsoft cluster dataset for us
import w2v_sub_matrix        #Contains the custom built word2vec substitution matrix


def load_data(raw_msr_fn):
    all_phrases = parse_microsoft.get_microsoft_phrases(raw_msr_fn)

    return all_phrases


def run_alignments():
    # Raw data
    # Provided by a colleague - available upon request
    raw_msr_fn = 'data/msr_paraphrase_data.txt'

    #Which w2v model you want to use
    w2v_bin_fn = 'data/memetracker-clusters-phrases.bin' #Use this if you want to train a word2vec vector space
    #w2v_bin_fn = 'data/GoogleNews-vectors-negative300.bin' #Use this if you want to use pre-trained w2v vector space

    phrasesX = load_data(raw_msr_fn)
    phrasesY = phrasesX.values()

    #Create substitution matrix object
    sub_matrix = w2v_sub_matrix.w2v_sub_matrix(w2v_bin_fn, 'msr')

    pqs = run_global_alignment.run_global_alignments(phrasesX, phrasesY, sub_matrix)

    run_global_alignment.print_priority_queues(pqs)


if __name__ == '__main__':
    run_alignments()