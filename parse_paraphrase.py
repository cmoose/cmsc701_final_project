import os.path
import pickle

def parse_cluster_data(fh):
    clusters = {}
    cur_cluster = {}
    cur_cluster_id = 0
    for line in fh:
        words = line.strip("\n").split("|||")
        print words[1], words[2]
        # New cluster found, so add cur_cluster to clusters, and start new
        if len(cur_cluster) > 0:
            clusters[cur_cluster_id] = cur_cluster
        cur_cluster = {'root': words[1].strip(" "), 'phrases': {}}
        cur_cluster_id = words[2].strip(" ")
        cur_cluster['phrases'][words[1].strip(" ")] = words[1].strip(" ")
        cur_cluster['phrases'][words[2].strip(" ")] = words[2].strip(" ")
        print cur_cluster

    # Finally, add last cluster to clusters
    if len(cur_cluster) > 0:
        clusters[cur_cluster_id] = cur_cluster

    return clusters

def load_ppdb_data(raw_ppdb_fn, clusters_pkl_fn):
    clusters = {}
    if os.path.isfile(clusters_pkl_fn):
        clusters = pickle.load(open(clusters_pkl_fn))
    else:
        # Process raw file
        clusters = parse_cluster_data(open(raw_ppdb_fn))
        pickle.dump(clusters, open(clusters_pkl_fn, 'wb'))

    return clusters


# Returns just the phrases, removing associations with clusters
def get_ppdb_phrases(clusters):
    all_phrases = []
    for cluster in clusters.values():
        for phrase in cluster['phrases'].values():
            all_phrases.append(phrase)

    return all_phrases


def write_phrases_to_file(phrases, phrases_fn):
    fhw = open(phrases_fn, 'wb')
    for phrase in phrases:
        fhw.write(phrase + '\n')

#print load_ppdb_data("data/ppdb-1.0-l-phrasal", "evaluation/data_ppdb.pkl")

