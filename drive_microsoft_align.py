# Drives the alignments of the microsoft phrase cluster data
#
# Author: Chris Musialek
# Date: Nov 2015
#

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_microsoft       #Loads the microsoft cluster dataset for us
import w2v_sub_matrix        #Contains the custom built word2vec substitution matrix
import os.path
import re


def load_data(raw_msr_fn):
    all_phrases = parse_microsoft.get_microsoft_phrases(raw_msr_fn)

    return all_phrases


def get_completed_phrases():
    l = os.listdir('./pkl')
    regex = re.compile('^([0-9].*)\.pkl') #all filenames of all digits plus .pkl
    compl = [int(regex.search(x).group(1)) for x in l if regex.search(x)]

    return compl


def run_alignments():
    # Raw data
    # Provided by a colleague - available upon request
    raw_msr_fn = 'data/msr_paraphrase_data.txt'

    phrasesX = load_data(raw_msr_fn)
    phrasesY = phrasesX.values()

    pqs = run_global_alignment.run_global_alignments(phrasesX, phrasesY, w2v_sub_matrix.word2vec_sub_matrix)

    run_global_alignment.print_priority_queues(pqs)


if __name__ == '__main__':
    run_alignments()