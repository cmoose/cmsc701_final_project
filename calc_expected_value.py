# Calculate expected value of a given substitution matrix
# E = SUM(p_i*p_j*S_ij)
# Author: Chris Musialek
# Date: Nov 2015
#
# Note: Running this takes a _very_ long time due to the size of the vocab.

#Get the full vocab of the data
import os.path
import pickle
import util
from gensim.models import Word2Vec
from collections import Counter

git_repo_path = os.path.dirname(os.path.realpath(__file__))
cornell_en_quotes_lemma_file = 'data/en_quotes_2008-08.lemma.txt'
vocab_pickle_file = 'pkl/vocab.en_quotes_2008-08.lemma.pkl'
w2v_bin_filename = 'data/en_quotes_2008-08.lemma.vectors.bin'

def build_vocab(filename):
    vocab = Counter()
    fh = open(os.path.join(git_repo_path, filename))
    for line in fh:
        tokens = line.split()
        for t in tokens:
            vocab[t] += 1
    return vocab


def create_pickle_vocab_file():
    '''Preprocessing to capture counts of tokens needed for p_i and p_j probabilities'''
    vocab = build_vocab(cornell_en_quotes_lemma_file)
    pickle.dump(vocab, open(os.path.join(git_repo_path, vocab_pickle_file), 'wb'))


#E = SIGMA(p_i*p_j*S_ij)
#Long and slow way
def calc_expected_value_naive(vocab_counts):
    model = Word2Vec.load_word2vec_format(os.path.join(git_repo_path,w2v_bin_filename), binary=True)
    E = 0
    for w_i, p_i in vocab_counts.items():
        for w_j, p_j in vocab_counts.items():
            if w_i == w_j:
                E += p_i*p_j #word match has s_ij=1
            else:
                try:
                    s_ij = model.similarity(w_i, w_j)
                    E += p_i*p_j*s_ij
                except KeyError:
                    E += p_i*p_j*-1 #s_ij=-1 for words not in word2vec matrix
    return E


def calc_expected_word2vec_words(P, count_of_counts, all_vocab_prob):
    model = Word2Vec.load_word2vec_format(os.path.join(git_repo_path,w2v_bin_filename), binary=True)
    all_vocab_prob.normalize()

    #Two sums: one where both words are in word2vec vocab
    #other when one word is in word2vec vocab, other is one of 6 counts
    s_model_vocab = set(model.vocab)
    s_all_vocab = set(all_vocab_prob.keys())
    s_not_word2vec_vocab = s_all_vocab.difference(s_model_vocab)

    E_oneonly = 0
    for word1 in s_model_vocab:
        for i in count_of_counts.keys():
            E_oneonly += all_vocab_prob[word1]*count_of_counts[i]*P[i]*-1

    E_both = 0
    #Slowest part - only naive section
    fhw = open('Expected_values_word2vec_words.txt', 'wb', 0)
    for word1 in s_model_vocab:
        E_word1 = 0
        print word1
        for word2 in s_model_vocab:
            joint_prob = all_vocab_prob[word1]*all_vocab_prob[word2] #P_i*P_j
            if word1 == word2:
                E_word1 += joint_prob #S_ij is 1 here because words match
            else:
                s_ij = model.similarity(word1, word2)
                E_word1 += joint_prob*s_ij
        #For resuming
        try:
            fhw.write(word1 + "\t" + str(E_word1) + "\n")
        except:
            print u'Error writing to file: word {0}'.format(word1)
        E_both += E_word1

    E = E_both + 2*E_oneonly
    return E


def get_non_word2vec_counts(all_vocab_counts):
    model = Word2Vec.load_word2vec_format(os.path.join(git_repo_path,w2v_bin_filename), binary=True)
    s_model_vocab = set(model.vocab)
    s_all_vocab = set(all_vocab_counts.keys())
    s_not_word2vec_vocab = s_all_vocab.difference(s_model_vocab)

    non_word2vec_vocab_counts = util.Counter() #counts of non word2vec vocab words
    for v in s_not_word2vec_vocab:
        non_word2vec_vocab_counts[v] = all_vocab_counts[v]
    counts_of_counts = Counter(non_word2vec_vocab_counts.values())

    count_probs = {}
    for i in counts_of_counts.keys():
        count_probs[i] = float(i)/sum(all_vocab_counts.values())

    return count_probs, counts_of_counts, non_word2vec_vocab_counts


def calc_expected_non_word2vec_words(P, count_of_counts):
    E = 0
    for i in count_of_counts.keys():
        for j in count_of_counts.keys():
            if i == j:
                #=((63837*63837)-63837)*p1*p1*-1 + 63837*p1*p1*1
                count = count_of_counts[i]
                E += ((count*count)-count)*P[i]*P[i]*-1 + count*P[i]*P[i]
            else:
                E += count_of_counts[i]*count_of_counts[j]*P[i]*P[j]*-1
    return E


def calc_expected_value(all_vocab_counts):
    #P=probabilities of each unique non-word2vec count (for each word)
    #count_of_counts = counts of each unique non-word2vec count
    #vocab_counts = dictionary of non word2vec tokens and their counts
    P, count_of_counts, non_word2vec_vocab_counts = get_non_word2vec_counts(all_vocab_counts)
    E_non = calc_expected_non_word2vec_words(P, count_of_counts)
    print "Non-word2vec expected value = {0}".format(E_non)

    E = calc_expected_word2vec_words(P, count_of_counts, all_vocab_counts)
    return E + E_non

if __name__ == '__main__':
    vocab_counts = pickle.load(open(os.path.join(git_repo_path, vocab_pickle_file)))
    E = calc_expected_value(vocab_counts)
    print E
