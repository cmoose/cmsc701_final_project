#Drives word2vec

#./word2phrase -train news.2012.en.shuffled-norm0 -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
#./word2vec -train news.2012.en.shuffled-norm1-phrase1 -output
# vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15

import subprocess


def run_word2vec(corpus_input_fn, w2v_bin_output_fn):
    cmd = ['bin/word2vec', '-train', corpus_input_fn,
           '-output', w2v_bin_output_fn, '-cbow 1', '-size 200', '-window 10', '-negative 25',
            '-hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15']
    print "Calling: " + " ".join(cmd)
    subprocess.call(cmd)


def run_word2phrase(input_fn, output_fn, iter):

    threshold = 200
    if iter > 0: threshold = 100

    cmd = ['bin/word2phrase', '-train', input_fn,
           '-output', output_fn, '-threshold', str(threshold), '-debug', '2']
    print "Calling: " + " ".join(cmd)
    subprocess.call(cmd)


def create_bin_file():
    corpus_basename = 'memetracker-clusters-phrases'
    corpus_fn = 'data/{0}'.format(corpus_basename)
    w2v_bin_output_fn = 'data/{0}.bin'.format(corpus_basename)

    # Run pipeline
    run_word2phrase(corpus_fn, corpus_fn + '-iter0', 0)
    run_word2phrase(corpus_fn + '-iter0', corpus_fn + '-iter1', 1)
    run_word2vec(corpus_fn + '-iter1', w2v_bin_output_fn)


if __name__ == '__main__':
    create_bin_file()