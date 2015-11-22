
# coding: utf-8

# # CMSC701 Final Project

# ## Document Similarity through Substitution Matrix Generation and Local Word Alignment

# ### Project Team
# - Sriram Karthik Badam (sbadam@umd.edu)
# - Chris Musialek (chris@musialek.org)
# - Deok Gun Park (intuinno@umd.edu)

# ### Introduction
# In this project, we propose to create a generalized substitution matrix that can be used to match and rank the similarity of documents within a corpus of text. This substitution matrix will be a |V| x |V| matrix representing the entire vocabulary of words (or words of some importance) from the text corpus. We will then evaluate this scoring system on our text corpus to understand its performance in aligning similar sets of text. During this project, we will explore two different approaches for generating the substitution matrix.
#
# ### Approaches
# #### Latent Semantic Analysis and Local Alignment
# One intuition we have is that similar words reside in the same semantic spaces. Latent Semantic Analysis is a technique using SVD to generate vectors representing the meanings of words, which can be used to build a multidimensional space of the words in the vocabulary, allowing us to calculate the distance between all words in the corpus. We think that these calculations could be a useful source of measures for the substitution matrix needed to produce the local alignment scores.
#
# Several LSA software packages exist that we think can learn reasonable word vectors to build the semantic space. They are listed in the References section below. Additionally, there are some manually-annotated word similarity data available (such as here) that we think will help us to evaluate the accuracy of our word vectors once we run LSA on our text.
#
# #### N-Grams and Local Alignment
# The Blosum or PAM matrices are developed on identical protein sequences to make matching relatively easy.  For example, PAM1 matrix represents one change per 100 amino acids.  This was derived from the analysis of accurately aligned homologous proteins. Similarly, we can apply this approach to generate a word substitution matrix for documents within a text corpus. However, one major difference between amino acids sequences and human language sentences is that the lexicon is much larger (e.g., orders of a few hundred thousand for English). This means finding accurately aligned long sequences of words in natural languages is less probable than amino acid/nucleotide sequences.
# In our second approach, we plan to explore trigram matching.  Between two trigrams (A, B), where  A = [a1 a2 a3] and B = [b1 b2 b3], if words within (a1, b1) and (a3, b3) pairs match, we can assume that b2 can be substituted with a2. By counting these a2 to b2 substitutions and normalizing them by the number of times a2 is observed in the trigrams (in the form T = [x a2 y]), we can estimate the probability of substitution between specific words, to generate the substitution matrix (pi->j). This matrix can be further used for the local alignment between the documents.
#
# #### Datasets
# We have several ideas for potential datasets, but could use some additional advice at this point. One type of text that we think would be interesting is events in news media. Often, journalistic articles contain snippets of quotes which may change over time. In addition, there are some great, already pre-processed, texts available on the word2vec website such as Wikipedia data, Google News data, and additional corpora. Lastly, we plan to reach out to Hanan Samet to see if their preprocessed Newsstand data may be available for us to use.
# Applications
# Document similarities based on edit distance are more effective for applications requiring context-sensitive document matching, where comparison of the sentence structures in the document is important (for example, plagiarism detection). In comparison, alternative methods based on topic modeling utilizing bag-of-words models can be more effective for estimating the similarities in the latent topic space, and are not ideal for plagiarism detection as they tend to lose the structural information.
#
# #### References
# ##### Papers solving similar problems:
# - http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
# - http://nlp.stanford.edu/~manning/papers/SocherHuangPenningtonNgManning_NIPS2011.pdf
#
# ##### Survey papers:
# Text Similarity http://research.ijcaonline.org/volume68/number13/pxc3887118.pdf
# String Matching http://knoesis.cs.wright.edu/faculty/pascal/pub/strings-iswc13.pdf
#
# ##### LSA Software:
# https://github.com/fozziethebeat/S-Space/wiki/GettingStarted
# http://infomap-nlp.sourceforge.net/
# http://edutechwiki.unige.ch/en/Latent_semantic_analysis_and_indexing#Software
# https://code.google.com/p/word2vec/
#

# ## Local Alignment using trigrams

# ### Load Library

# In[26]:

import nltk
from nltk.collocations import *
import time
import cPickle as pickle
from nltk.corpus import stopwords
from itertools import combinations_with_replacement


# ### Data Preprocessing
#
# Here I will parse the data and remove tokens that do not occur more than unigram threshold times.
#
# #### Hyper parameters
#  - thresholdUnigram = 100
#

# In[27]:

#######################
# Tuning parameters for program
thresholdUnigram = 10
specialSpacer = 'IntuinnoSpecialSpacing'

#######################


# In[28]:


trigramFD = nltk.FreqDist()

for fileloop in range(114):


    start = time.time()

    # file = open('en_quotes_2008-08.lemma.txt')
    file = open('split' + str(fileloop).zfill(3))
    t = file.read()


    tokens = nltk.word_tokenize(t)
    lowerTokens = [w.lower() for w in tokens]
    unigram = nltk.FreqDist(tokens)
    frequentTokens = {k:v for k,v in unigram.items() if v > thresholdUnigram and k not in stopwords.words('english')}

    # reducedText = [k if k in frequentTokens else specialSpacer  for k in tokens]


    trigramFinder = TrigramCollocationFinder.from_words(lowerTokens)

    trigramFinder.apply_ngram_filter( lambda w1, w2, w3: w2 not in frequentTokens )
    # trigramFinder.apply_freq_filter(2)

    trigramFD |=  trigramFinder.ngram_fd
    end = time.time()

    print end - start
    print fileloop


# ### Count aligned word distribution
#
# Here I will iterate over all the trigram and create a aligned freq distribution

# In[29]:

alignedWords = {}

for word, frequency in trigramFD.iteritems():
    if not (word[0], word[2]) in alignedWords:
        alignedWords[(word[0], word[2])] = nltk.FreqDist()

    alignedWords[(word[0], word[2])][word[1]] += frequency


# In[32]:

substitutionCount = nltk.FreqDist()
possibleCount = nltk.FreqDist()

countAllCombinatinations = 0

for k, v in alignedWords.iteritems():

    if len(v) > 1:
        for w1, w2 in combinations_with_replacement(v,2):
            if w1 == w2 and v[w1] != 1:
                substitutionCount[(w1,w2)] += v[w1] * (v[w1]-1) / 2
                countAllCombinatinations += v[w1] * (v[w1]-1) /2
            else:
                substitutionCount[(w1,w2)] += v[w1] * v[w2]
                countAllCombinatinations += v[w1] * v[w2]



# In[33]:

q = nltk.FreqDist()

wordlist = nltk.FreqDist()
for w1, w2 in substitutionCount:
    q[(w1,w2)] = 1.0 * substitutionCount[(w1,w2)] /countAllCombinatinations
    wordlist[w1] += 1
    wordlist[w2] += 1


def sumFreqDist(fd):
    result = 0
    for a in fd:
        result += fd[a]

    return result


print sumFreqDist(q)
print sumFreqDist(substitutionCount)
print sumFreqDist(trigramFD)


prob = {}
for w_i, f_i  in wordlist.iteritems():
    sumRemainder = 0
    for w_j, f_j in wordlist.iteritems():
        if w_i < w_j:
            sumRemainder += q[(w_i,w_j)]
        elif w_i > w_j:
            sumRemainder += q[(w_j,w_i)]


    prob[w_i] = q[(word, word)] + sumRemainder / 2

expectedFrequency = {}


newWordlist = []


for w, f in wordlist.iteritems():
    if prob[w] != 0:
        newWordlist.append(w)

for w_i in newWordlist:
    for w_j in newWordlist:
        if w_i != w_j:
            expectedFrequency[(w_i, w_j)] = prob[w_i] * prob[w_j] * 2

        else:
            expectedFrequency[(w_i, w_j)] = prob[w_i] * prob[w_i]



score = {}

for w_i in newWordlist:
    for w_j in newWordlist:
        if expectedFrequency[(w_i,w_j)] == 0:
            score[(w_i,w_j)] = 0
            score[(w_j, w_i)] = 0
        elif w_i < w_j:
            score[w_i,w_j] =  q[(w_i,w_j)] / expectedFrequency[(w_i,w_j)]
        elif w_i > w_j:
            score[w_i,w_j] =  q[(w_j,w_i)] / expectedFrequency[(w_i,w_j)]


print len(newWordlist)

print "I am ending"

pickle.dump(score, open('score.pickle','wb'))