## Computational Genomics Final Project

## Prerequisites for memetracker cluster dataset
1. python
2. python module gensim
3. (if not running on a mac) word2vec and word2phrase binaries

## Prerequisites for memetracker raw dataset
1. python
2. python langid module (detects English phrases, used for preprocessing)
3. python module gensim
4. maven (used for preprocessing)
5. java (used for preprocessing)
6. (if not running on a mac) word2vec and word2phrase binaries

## Running global alignment
- Download Memetracker cluster dataset into data/ directory
- run `python drive_memcluster_align.py`
- top aligned phrases are stored as pickle files in data/

## Datasets
### Memetracker
Download from http://www.memetracker.org/data.html 
- Memetracker cluster dataset link: [link](http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/clust-qt08080902w3mfq5.txt.gz). 
- Raw dataset uses the [Aug 2008](http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/quotes_2008-08.txt.gz) file.
### Penn Paraphrase Database
Download from http://www.cis.upenn.edu/~ccb/ppdb/
### Microsoft Research Paraphrases
Available upon request

## Preprocessing raw memetracker data
- in nlp/ directory, run `mvn install`
- Modify preprocess_cornell_quotes.py to point to downloaded dataset file location
- run `python preprocess_cornell_quotes.py`
- This produces the lemmatized file of quotes


