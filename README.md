## Computational Genomics Final Project

## Prerequisites
1. maven
2. java
3. python
4. python langid module

## Dataset
Download from http://www.memetracker.org/data.html. Currently using the [Aug 2008](http://snap.stanford.edu/data/d/quotes/Old-UniqUrls/quotes_2008-08.txt.gz) file.

## Preprocessing data
- in nlp/ directory, run `mvn install`
- Modify preprocess_cornell_quotes.py to point to downloaded dataset file location
- run `python preprocess_cornell_quotes.py`
- This produces the lemmatized file of quotes
