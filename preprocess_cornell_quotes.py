# Does preprocessing work on the cornell quotes text
# Author: Chris Musialek
# Date: Nov 2015
#

import gzip
import langid
import subprocess
import os.path

cornell_gzip_quotes_file = '/Users/chris/Downloads/quotes_2008-08.txt.gz'
cornell_en_quotes_file = 'en_quotes_2008-08.txt'
cornell_en_quotes_lemma_file = 'en_quotes_2008-08.lemma.txt'
git_repo_path = os.path.dirname(os.path.realpath(__file__))

def filter_english_quotations():
    if not os.path.isfile(cornell_en_quotes_file):
        fh = gzip.open(cornell_gzip_quotes_file)
        fhw = open(cornell_en_quotes_file, 'wb')

        #Go through each line, pull out the English quotes
        #and write them to the new file
        for line in fh:
            if line.startswith('Q'):
                text = line.split('\t')[1]
                lang = langid.classify(text)[0]
                if lang == 'en':
                    fhw.write(text)


def lemmatize_english_quotations():
    if not os.path.isfile(cornell_en_quotes_lemma_file):
        #Run CoreNLP code in nlp/ to lemmatize
        _cp = '{0}/nlp/target/nlp-1.0-SNAPSHOT.jar:{0}/nlp/target/lib/*'.format(git_repo_path)
        _inputfile = '{0}'.format(os.path.join(git_repo_path, cornell_en_quotes_file))
        _outputfile = '{0}'.format(os.path.join(git_repo_path, cornell_en_quotes_lemma_file))
        _main = 'edu.umd.CoreNLPprocessor'
        cmd = ["java", "-cp", _cp, _main, "-inputfile", _inputfile, "-outputfile", _outputfile]
        print " ".join(cmd)

        subprocess.call(cmd)


if __name__ == '__main__':
    filter_english_quotations()
    lemmatize_english_quotations()

