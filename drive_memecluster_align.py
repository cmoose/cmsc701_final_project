# Drives the alignments of the memetracker phrase cluster data
#
# Author: Chris Musialek
# Date: Nov 2015

import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_memetracker     #Loads the data for us

def load_data():
    return parse_memetracker.load_memetracker_data()

def run_alignment():
    clusters = load_data()
    all_phrases = parse_memetracker.get_memtracker_phrases(clusters)
    parse_memetracker.write_phrases_to_file(all_phrases)


if __name__ == '__main__':
    run_alignment()