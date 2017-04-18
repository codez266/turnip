#from gensim.parsing.preprocessing import STOPWORDS
#from gensim.parsing import PorterStemmer
import string
import re
import sys

def processToken(token, stemmer):
	if token not in STOPWORDS and token not in string.punctuation:
		return token.lower()
	else:
		return ""


def preprocess(file, out):
	#stemmer = PorterStemmer()
	f = open(file)
	outp = open(out, 'w')
	for line in f:
		res = ""
		res = line.strip()
		#for word in line.split():
		#    w = processToken(word, stemmer)
		#    if w != "":
		#        #w = re.sub('\[([^ \[\]\|]+)\|([^\[\]\|]+)\]', r'\1', w)
		#        str = str + w + " "
		res = re.sub('\[([^ \[\]\|]+)\|([^\[\]\|]+)\]', r'\2', res)
		outp.write(res + "\n")
	f.close()
	outp.close()


if __name__ == '__main__':
	inp, out = sys.argv[1:3]
	preprocess(inp, out)
