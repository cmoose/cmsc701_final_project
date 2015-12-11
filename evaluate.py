
import run_global_alignment  #Contains Needleman-Wunsch algorithm
import parse_memetracker     #Loads the memetracker cluster dataset for us
import w2v_sub_matrix        #Contains the custom built word2vec substitution matrix
import train_word2vec              #Word2vec preprocessing driver
import os.path
import random
import re
import cPickle
import gzip
import sys
import numpy
import blosum

# Load raw gz data, save as pkl file, return cluster datastructure
def load_data(raw_gz_fn, clusters_pkl_fn):
    return parse_memetracker.load_memetracker_data(raw_gz_fn, clusters_pkl_fn)


#Load the raw phrases (built from word2phrase iterations)
def load_data1(w2v_phrases_fn):
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
    train_word2vec.create_bin_file(w2v_basename)

    # We return the two filenames needed to run the alignments
    return w2v_basename + '.bin', w2v_basename + '-final'


def get_completed_phrases():
    l = os.listdir('./pkl')
    regex = re.compile('^([0-9].*)\.pkl') #all filenames of all digits plus .pkl

    compl = [int(regex.search(x).group(1)) for x in l if regex.search(x)]

    return compl

# For set of N phrases, we select a target phrase and its m paraphrase.
# Among the remainder, we select k random phrases.
# We calculate global alignment score with m paraphrase and k random phrases.
# We check whether the global alignment score is higher with paraphrases and low with random phrases.
#
# Parameters
# --------------------
# dataset = 'PPDB', 'MEME', or 'MS'
# subMatrix = 'w2v' or 'blosum'
# numTestCase - Total number of Trial
# numPhrases - Group Size for Sample
# topK - If the truth make it in TopK words, it is Success else Failure
def run_evaluation(dataset, subMatrix, numTestCase=10, numPhrases=5000, topK=10):
    # Files required to run evaluation
    data_ppdb_fn = 'evaluation/data_ppdb.pkl'
    data_meme_fn = 'evaluation/data_meme.pkl'
    data_ms_fn = 'evaluation/data_ms.pkl'

    # Load dataset
    if dataset == 'MEME':
        clusters = cPickle.load(open(data_meme_fn,'rb'))
    elif dataset == 'PPDB':
        clusters = cPickle.load(open(data_ppdb_fn,'rb'))
    elif dataset == 'MS':
        clusters = cPickle.load(open(data_ms_fn,'rb'))
    else:
        sys.exit( 'Unsupported dataset.  Choose from "MEME","PPDB", or "MS"')


    # Load Substitution Matrix
    if subMatrix == 'w2v':
        submat = w2v_sub_matrix.word2vec_sub_matrix
    elif subMatrix == 'blosum':
        submat = blosum.blosum_sub_matrix

    evaluate_n_times(clusters, submat, numTestCase, numPhrases, topK)

def evaluate_n_times(clusters, subMatrix, numTestCase, numPhrases, topK):
    result = []
    for i in range(numTestCase):
        targetSentence, clusterPhrases, randomPhrases = sampleData(clusters, numPhrases)
        evaluationScores = get_evaluation_score(subMatrix, targetSentence, clusterPhrases, randomPhrases, topK)
        result.append(evaluationScores)
    get_average_score(result)

def get_average_score(scoresList):
    clusterScores = [w['avgClusterScore'] for w in scoresList]
    randomScores = [w['avgRandomScore'] for w in scoresList]
    rankings = [w['avgRanking'] for w in scoresList]
    TP = numpy.sum([w['TP'] for w in scoresList])
    TN = numpy.sum([w['TN'] for w in scoresList])
    FP = numpy.sum([w['FP'] for w in scoresList])


    avgClusterScore = numpy.mean(clusterScores)
    avgRandomScore = numpy.mean(randomScores)
    avgRanking = numpy.mean(rankings)

    precisions = TP/float(TP + FP)
    recalls = TP/float(TP + TN)
    fScores = 2 * precisions * recalls / (precisions + recalls)

    print '--------------------------------------'
    print ' Result for {0} tests'.format(len(scoresList))
    print '--------------------------------------'
    print 'Alignment Scores for similar phrases:\t' + str(avgClusterScore)
    print 'Alignment Scores for random phrases:\t' + str(avgRandomScore)
    print 'Average Ranking for similar phrases:\t' + str(avgRanking)
    print 'Average Precision:\t' + str(precisions)
    print 'Average Recalls:\t' + str(recalls)
    print 'Average fScores:\t' + str(fScores)



def sampleData(clusters, numPhrases):

    targetKey = random.choice(clusters.keys())
    targetCluster = clusters[targetKey]
    targetPhrase = targetCluster['root']
    clusterPhrases = targetCluster['phrases'].values()

    randomPhrases = []

    for i in range(numPhrases - len(clusterPhrases)):
        randomKey = random.choice(clusters.keys())
        randomCluster = clusters[randomKey]
        randomPhraseKey = random.choice(randomCluster['phrases'].keys())
        randomPhrases.append(randomCluster['phrases'][randomPhraseKey])

    return targetPhrase, clusterPhrases, randomPhrases

def get_evaluation_score(subMatrix, targetSentence, clusterPhrases, randomPhrases, topK):
    clusterScores = run_global_alignment.eval_global_align(targetSentence, clusterPhrases, subMatrix)
    randomScores = run_global_alignment.eval_global_align(targetSentence, randomPhrases, subMatrix)

    averageClusterScore = numpy.mean(clusterScores)
    averageRandomScore = numpy.mean(randomScores)

    ranking = get_ranking(clusterScores, randomScores)

    averageRanking = numpy.mean(ranking)
    TP, TN, FP, precision, recall, fScore = get_contigency_score(ranking, len(clusterPhrases) + len(randomPhrases), topK)

    print 'Scores for ' + str(targetSentence) + ':'
    print 'Average Score for Cluster Phrases: ' + str(averageClusterScore),
    print '\t Average Score for Random Phrases: ' + str(averageRandomScore),
    print '\t Average Ranking for Cluster Phrases: ' + str(averageRanking),
    print 'Contigency Score for Top' + str(topK)
    print 'Precision: ' + str(precision),
    print '\tRecall: ' + str(recall),
    print '\tF-Score: ' + str(fScore)

    return {'avgClusterScore': averageClusterScore,
            'avgRandomScore': averageRandomScore,
            'avgRanking': averageRanking,
            'p': precision,
            'r': recall,
            'f': fScore,
            'TP': TP,
            'TN': TN,
            'FP': FP}

def get_ranking(clusterScores, randomScores):
    sortedCluster = sorted(clusterScores)
    sortedRandom = sorted(randomScores)
    result = []
    for i in range(len(clusterScores)+len(randomScores)):
        if len(sortedCluster) == 0:
            break
        if len(sortedRandom) == 0:
            sys.exit('Something wrong in get_ranking')
        if sortedCluster[0] >= sortedRandom[0]:
            result.append(i+1)
            sortedCluster.pop(0)
        else:
            sortedRandom.pop(0)

    if len(clusterScores) != len(result):
        sys.exit('Something wrong in get_ranking')
    return result

def get_contigency_score(ranking, n, topK):
    TP = len([ w for w in ranking if w <= topK])
    TN = len(ranking) - TP
    FP = len(ranking) - TP
    FN = n - len(ranking) - TN

    Precision = TP / float(TP + FP)
    Recall = TP / float(TP + TN)

    fScore = Precision * Recall / (Precision + Recall)

    return TP, TN, FP, Precision, Recall, fScore


if __name__ == '__main__':
    run_evaluation('MEME', 'w2v', numTestCase=10, numPhrases=100, topK=10 )


