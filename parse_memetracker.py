import re
import pickle
import os.path
import gzip

# clusters = {'<ClId>': {}}
# cluster = {'root': '<Root>', 'phrases': {'<QtId>': {}}}
# phrase = {'<QtStr>'}
def parse_cluster_data(fh):
    lvl1regex = re.compile('^\d+\t\d+\t(.*)\t(\d+)')
    lvl2regex = re.compile('^\t\d+\t\d+\t(.*)\t(\d+)')
    clusters = {}
    cur_cluster = {}
    cur_cluster_id = 0
    for line in fh:
        if lvl1regex.search(line):
            match = lvl1regex.search(line)
            # New cluster found, so add cur_cluster to clusters, and start new
            if len(cur_cluster) > 0:
                clusters[cur_cluster_id] = cur_cluster
            cur_cluster = {'root': match.group(1), 'phrases': {}}
            cur_cluster_id = match.group(2)
        elif lvl2regex.search(line):
            match = lvl2regex.search(line)
            cur_cluster['phrases'][match.group(2)] = match.group(1)
    # Finally, add last cluster to clusters
    if len(cur_cluster) > 0:
        clusters[cur_cluster_id] = cur_cluster

    return clusters


def load_memetracker_data():
    clusters = {}
    if os.path.isfile('pkl/memetracker-clusters.pkl'):
        clusters = pickle.load(open('pkl/memetracker-clusters.pkl'))
    else:
        # Process raw file
        # Dwnld from http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz
        fh = gzip.open('data/clust-qt08080902w3mfq5.txt.gz')
        clusters = parse_cluster_data(fh)
        pickle.dump(clusters, open('data/memetracker-clusters.pkl', 'wb'))

    return clusters


# Returns just the phrases, removing associations with clusters
def get_memtracker_phrases(clusters):
    all_phrases = []
    for cluster in clusters.values():
        for phrase in cluster['phrases'].values():
            all_phrases.append(phrase)

    return all_phrases


def write_phrases_to_file(phrases):
    fhw = open('data/memetracker-clusters-phrases.txt', 'wb')
    for phrase in phrases:
        fhw.write(phrase + '\n')

