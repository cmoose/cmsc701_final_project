# Utility module for loading microsoft research data
#
# Author: Chris Musialek
# Date: Nov 2015
#

def parse_microsoft_train(fh):
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


def load_microsoft_train(raw_fn):
    clusters = {}
    fh = open(raw_fn)
    clusters = parse_microsoft_train(fh)

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