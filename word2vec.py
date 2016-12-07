#!/usr/bin/env python

import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
import re

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        for line in open(self.filename, 'r'):
            textline = line.lower()
            text=re.sub(r'\[(.*?)\|(.*?)\]',r'\1',textline)
            text=re.sub(r'[\[\]\(\)\:"\.;\',]',r'', text)
            yield text.split()

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

    model = Word2Vec(MySentences(inp), size = 400, window = 8, min_count = 5,
                     workers = multiprocessing.cpu_count())

    model.init_sims(replace=True)

    model.save(outp)

