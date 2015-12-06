## Computational Genomics Final Project - Document Similarity

## Prerequisites for word2vec approach
1. python
2. python module gensim
3. (if not running on a mac) compiled [word2vec](https://code.google.com/p/word2vec/) and word2phrase binaries

## Running a global alignment using word2vec based substitution matrix
1. Download Memetracker cluster dataset into data/ directory.
1. Train the word2vec vectors or use Google's pre-trained model
  - Download Google's pre-trained model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) into data/ directory.
  - OR run `python word2vec.py` to create one from memetracker-cluster-dataset
1. If using Google's pre-trained model:
  1. gunzip downloaded file in data/ directory
  2. open `w2v_sub_matrix.py` and replace custom_w2v_bin_fn with google_w2v_bin_fn in line 19.
- run `python drive_memecluster_align.py` to create and print alignments
  - top aligned phrases per phrase will print aligned alongside alignment score
  - top aligned phrases are stored as pickle files in data/

## Datasets
### 1. [Memetracker](http://www.memetracker.org/data.html)
- Memetracker cluster dataset can be downloaded [here](http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz). 

### 2. Penn Paraphrase Database
- Download from http://www.cis.upenn.edu/~ccb/ppdb/

### 3. Microsoft Research Paraphrases
- Available upon request


## Other Processing
### Prerequisites for memetracker raw dataset (not used in report)
1. python
2. python langid module (detects English phrases, used for preprocessing)
3. python module gensim
4. maven (used for preprocessing)
5. java (used for preprocessing)
6. (if not running on a mac) word2vec and word2phrase binaries

### Preprocessing raw memetracker data (not used in report)
- Download [raw phrase dataset](http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/quotes_2008-08.txt.gz)
- copy dataset into data/ directory
- cd to nlp/ directory, run `mvn install`
- run `python preprocess_cornell_quotes.py` (this actually runs a custom implementation of Stanford's CoreNLP)
  - This produces the lemmatized file of quotes
- run `python word2vec.py`




