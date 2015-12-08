# Drives word2vec binaries from within python.
# Used for training the word2vec model, which then is
# used to drive the sub matrix in alignments.
#
# Author: Chris Musialek
# Date: Nov 2015
#

# Examples of commands called
#./word2phrase -train news.2012.en.shuffled-norm0 -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
#./word2vec -train news.2012.en.shuffled-norm1-phrase1 -output
# vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15

import subprocess
import os.path
import drive_memecluster_align

# Calls word2vec binary
# @input String corpus_input_fn - input filename
# @input String w2v_bin_output_fn - output filename
def run_word2vec(corpus_input_fn, w2v_bin_output_fn):
    cmd = ['bin/word2vec', '-train', corpus_input_fn,
           '-output', w2v_bin_output_fn, '-cbow', '1', '-size', '200', '-window', '10', '-negative', '25',
            '-hs', '0', '-sample', '1e-5', '-threads', '20', '-binary', '1', '-iter', '15']
    print "Calling: " + " ".join(cmd)
    subprocess.call(cmd)
    print "Created new word2vec model...{0}".format(w2v_bin_output_fn)


# Calls word2phrase binary
# @input String input_fn - input filename
# @input String output_fn - output filename
# @input Integer threshold - threshold
def run_word2phrase(input_fn, output_fn, threshold):

    cmd = ['bin/word2phrase', '-train', input_fn,
           '-output', output_fn, '-threshold', str(threshold), '-debug', '2']
    print "Calling: " + " ".join(cmd)
    subprocess.call(cmd)


# Creates the binary file
# @input String corpus_basename - basename of the corpus text file
def create_bin_file(corpus_basename):
    corpus_fn = 'data/{0}.txt'.format(corpus_basename)
    w2v_bin_output_fn = 'data/{0}.bin'.format(corpus_basename)
    if not os.path.isfile(corpus_fn):
        if os.path.isfile('data/clust-qt08080902w3mfq5.txt.gz'):
            drive_memecluster_align.do_prep_work_load(corpus_basename, 'data/clust-qt08080902w3mfq5.txt.gz')
        else:
            print "ERROR: download memetracker cluster dataset into data/ directory first, then run this again."
            exit(1)

    # Run pipeline
    run_word2phrase(corpus_fn + '.txt', corpus_fn + '-int', 200)
    run_word2phrase(corpus_fn + '-int', corpus_fn + '-final', 100)
    run_word2vec(corpus_fn + '-final', w2v_bin_output_fn)


if __name__ == '__main__':
    corpus_basename = 'memetracker-clusters-phrases' #basename of the files that will be created
    create_bin_file(corpus_basename)