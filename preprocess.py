from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import PorterStemmer
import string
import re
import sys

def processToken(token, stemmer):
    if token not in STOPWORDS and token not in string.punctuation:
        return token.lower()
    else:
        return ""


def preprocess(file, out):
    stemmer = PorterStemmer()
    f = open(file)
    outp = open(out, 'w')
    for line in f:
        str = ""
        for word in line.split():
            w = processToken(word, stemmer)
            if w != "":
                #w = re.sub('\[([^ \[\]\|]+)\|([^\[\]\|]+)\]', r'\1', w)
                str = str + w + " "
        str = re.sub('\[([^ \[\]\|]+)\|([^\[\]\|]+)\]', r'\1', str)
        outp.write(str + "\n")
    f.close()
    outp.close()


if __name__ == '__main__':
    inp, out = sys.argv[1:3]
    preprocess(inp, out)
