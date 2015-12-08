# Utility module for loading microsoft research data.
# Used along side drive_microsoft_align.py to run alignments
#
# Author: Chris Musialek
# Date: Nov 2015
#

import pickle, os

def parse_microsoft_clusters(fh):
    phrases = []
    fh.next() #skip header
    for line in fh:
        l = line.split('\t')
        qual = l[0]
        id1 = l[1]
        id2 = l[2]
        phr1 = l[3].strip()
        phr2 = l[4].strip()
        paraphr = {'qual': qual, 'id1': id1, 'id2': id2, 'phrase1': phr1, 'phrase2': phr2}
        phrases.append(paraphr)
    return phrases

def parse_cluster_data(fh):
    clusters = {}
    cur_cluster = {}
    cur_cluster_id = 0
    for line in fh:
        words = line.strip("\n").split("\t")
        print words[3].strip(), words[4].strip()
        # New cluster found, so add cur_cluster to clusters, and start new
        if len(cur_cluster) > 0:
            clusters[cur_cluster_id] = cur_cluster
        cur_cluster = {'root': words[3].strip(" "), 'phrases': {}}
        cur_cluster_id = words[4].strip(" ")
        cur_cluster['phrases'][words[3].strip(" ")] = words[3].strip(" ")
        cur_cluster['phrases'][words[4].strip(" ")] = words[4].strip(" ")
        print cur_cluster

    # Finally, add last cluster to clusters
    if len(cur_cluster) > 0:
        clusters[cur_cluster_id] = cur_cluster

    return clusters

# All data
def get_microsoft_phrases(raw_fn):
    phrases = {}
    fh = open(raw_fn)
    fh.next() #skip header
    for line in fh:
        l = line.split('\t')
        _id = l[0]
        phr = l[1]
        phrases[_id] = [x.strip() for x in phr.strip().split()]
    return phrases


def load_microsoft_clusters(raw_fn):
    clusters = get_microsoft_phrases(open(raw_fn))
    return clusters


def load_microsoft_clusters2(raw_fn, clusters_pkl_fn):
    clusters = {}
    if os.path.isfile(clusters_pkl_fn):
        clusters = pickle.load(open(clusters_pkl_fn))
    else:
        # Process raw file
        clusters = parse_cluster_data(open(raw_fn))
        pickle.dump(clusters, open(clusters_pkl_fn, 'wb'))

    return clusters

# Returns just the phrases, removing associations with clusters
def get_microsoft_train_phrases(clusters):
    all_phrases = {}
    for cluster in clusters:
        id1 = cluster['id1']
        id2 = cluster['id2']
        all_phrases[id1] = cluster['phrase1']
        all_phrases[id2] = cluster['phrase2']

    return all_phrases

#load_microsoft_clusters2("data/msr_paraphrase_clusters.txt", "evaluation/data_ms.pkl")
