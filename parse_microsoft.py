import os.path
import pickle

def parse_microsoft_data(fh):
    phrases = []
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


def load_microsoft_data(raw_fn, clusters_pkl_fn):
    clusters = {}
    if os.path.isfile(clusters_pkl_fn):
        clusters = pickle.load(open(clusters_pkl_fn))
    else:
        # Process raw file
        fh = open(raw_fn)
        clusters = parse_microsoft_data(fh)
        pickle.dump(clusters, open(clusters_pkl_fn, 'wb'))

    return clusters


# Returns just the phrases, removing associations with clusters
def get_microsoft_phrases(clusters):
    all_phrases = {}
    for cluster in clusters:
        id1 = cluster['id1']
        id2 = cluster['id2']
        all_phrases[id1] = cluster['phrase1']
        all_phrases[id2] = cluster['phrase2']

    return all_phrases