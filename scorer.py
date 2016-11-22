import re
import logging
import mwparserfromhell as mwparser
import mwapi
import os
import urllib
import math
from nltk import PorterStemmer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class Scorer(object):
    def __init__(self, topics, pairs, mode):
        self.topics = topics
        self.pairs = pairs
        self.mode = mode
        self.persons = {}
        for pair in self.pairs:
            if pair[0] not in self.persons:
                self.persons[pair[0]] = {mode:[pair[1]], 'text':""}
            else:
                self.persons[pair[0]][mode].append(pair[1])

    def score(self):
        pass

    def writeScore(self, out):
        outp = open(out, 'w')
        for line in self.pairs:
            l = line
            outp.write(l[0] + "\t" + l[1] + "\t" + str(l[2]) + "\n")
        outp.close()

    def getTopicData(self, ind_file):
        ent = {}
        with open(ind_file) as f:
            for line in f:
                p = line.strip().split("\t")
                topic = p[0]
                words = p[2].split(",")
                if topic in self.topics:
                    ent[topic] = words
        return ent
    
    @staticmethod
    def getTopics(data_file):
        topics = []
        with open(data_file) as f:
            for line in f:
                topics.append(line.strip())
        return topics

    @staticmethod
    def getPersonPairs(data_file):
        pairs = []
        with open(data_file) as f:
            for line in f:
                l = line.strip().split("\t")
                pairs.append([l[0],l[1]])
        return pairs

    @staticmethod
    def getWikipediaTexts(data_file, persons):
        i = 0
        with open(data_file) as f:
            for line in f:
                l = line.strip("\n").split("\t")
                if l[0] in persons:
                    persons[l[0]]['text'] = l[1]
                    i = i + 1
                if i == len(persons):
                    break
        return persons

    @staticmethod
    def getWikipediaTexts2(persons):
        session = mwapi.Session("https://en.wikipedia.org", user_agent = "test" )
        for p in persons:
            if not os.path.isfile('articles/'+p):
                try:
                    logger.info("Requesting %s", p)
                    page = Scorer.fetchArticle(p, session)
                    persons[p]['text'] = page[p]
                    f = open('articles/'+p, 'w')
                    logger.info("Writing back: %s", p)
                    f.write(page[p])
                    f.close()
                except Exception as e:
                    print("Failed for:"+p)
                    logger.error("Falie for: %s\n %s", p, str(e))
            else:
                logger.info("Reading local: %s", p)
                f = open('articles/'+p)
                persons[p]['text'] = f.read()
                f.close()
        return persons
    
    @staticmethod
    def fetchArticle(title, session):
        pagecontent = {}
        doc = session.get(action="query", prop="revisions", rvprop="content", titles=title)
        pages = doc["query"]["pages"]
        for key, page in pages.items():
            text = page["revisions"][0]["*"]
            pagecontent[page["title"]] = Scorer.process_sentence(str(mwparser.parse(text)))
        return pagecontent

    @staticmethod
    def repl(m):
        return ''
    
    @staticmethod
    def process_sentence(sentence):
        # a heavenly regex, replaces mentions with actual names!
        t = re.sub(r"\[(.*?)\|(.*?)\]",r"\1",sentence)
        t = re.sub(r"[\(\)\.\[\]%<>',\":;0-9]", " ", t)
        t = re.sub(r"http.*", "",t)
        t = re.sub(r"\-", " ", t)
        t = re.sub(r"[^a-zA-Z_ \n]", "", t, flags=re.UNICODE)
        t = t.strip()
        return t

    @staticmethod
    def maplog(list):
        max = -1
        mapped = []
        for s in list:
            if max < s:
                max = s
        for i, s in enumerate(list):
            if max != 0:
                if s > 1:
                    mapped.append(s / max)
                    mapped[i] = mapped[i] * (2**7)
                    mapped[i] = int(math.log(mapped[i], 2))
                else:
                    mapped.append(0)
            else:
                mapped.append(s)
        return mapped

    @staticmethod
    def maplin(list):
        max = -1
        mapped = []
        for s in list:
            if max < s:
                max = s
        for i, s in enumerate(list):
            if max != 0:
                if s > 0:
                    mapped.append(s / max)
                    mapped[i] = int(mapped[i] * (7))
                else:
                    mapped.append(0)
            else:
                mapped.append(s)
        return mapped

class CountScorer(Scorer):
    def __init__(self, topics, pairs, mode):
        super().__init__(topics, pairs, mode)
   
    def testExtract(self):
        self.persons = Scorer,getWikipediaTexts('../persons3', self.persons)
        topicwords = Scorer.getTopicData('data/indicators-f')

    def score(self):
        stemmer = PorterStemmer()
        self.persons = Scorer.getWikipediaTexts2(self.persons)
        dummy = ""
        topicwords = self.getTopicData('data/indicators-f')
        i = 0
        for pair in self.pairs:
            per = pair[0]
            #print(dummy,",",per)
            prof = pair[1]
            if dummy != per:
                proflist = self.persons[per][self.mode]
                scorelist = []
                for p in proflist:
                    #print(p)
                    if p in topicwords:
                        words = [stemmer.stem(x.lower()) for x in topicwords[p]]
                        #print(words)
                        #print("calculating for:", per, p)
                        text = self.persons[per]['text']
                        score = 0
                        for w in text.split():
                            if stemmer.stem(w.lower()) in words:
                                score = score + 1
                        #print(score)
                        scorelist.append(score)
                    else:
                        scorelist.append(0)
                # all professions scored
                scorelist = Scorer.maplog(scorelist)
                # assign the scores to persons pair
                for j in range(i, i + len(scorelist)):
                    #print("Scoring %s, %s:%d",self.pairs[j][0], self.pairs[j][1],scorelist[j-i])
                    self.pairs[j].append(scorelist[j-i])
            dummy = per
            i = i + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input directory", action="store_true")
    parser.add_argument("input", help="Input directory value")

    parser.add_argument("-o", help="Output directory", action="store_true")
    parser.add_argument("output", help="Output directory value")
    args = parser.parse_args()

    input = ""
    output = ""
    if args.i:
        input = args.input
    if args.o:
        output = args.output

    topics = Scorer.getTopics(input+'/professions')
    pairs = Scorer.getPersonPairs(input+'/profession.train')
    sc = CountScorer(topics, pairs, 'profession')
    sc.score()
    sc.writeScore(output+'/profession.out')
if __name__ == '__main__':
    main()
