#!/usr/bin/env python

import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import PorterStemmer
import string


def processToken(token, stemmer):
    if token not in STOPWORDS and token not in string.punctuation:
        return stemmer.stem(token.lower())
    else:
        return ""


def preprocess(file, logger):
    logger.info("starting preprocess...")
    stemmer = PorterStemmer()
    f = open(file)
    lines = []
    for line in f:
        str = ""
        for word in line.split():
            w = processToken(word, stemmer)
            if w != "":
                str = str + w + " "
        lines.append(str.strip())
    f.close()
    logger.info("ending preprocess...")
    return lines


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp), size = 400, window = 5, min_count = 5,
                     workers = multiprocessing.cpu_count())

    model.init_sims(replace=True)

    model.save(outp)

