import re
import pdb
import logging
import mwparserfromhell as mwparser
import mwapi
import os
import urllib
import math
import inflect
import pickle
import json, requests
from gensim.models import Word2Vec
from nltk import PorterStemmer
from gensim.models import Word2Vec
import argparse
import ranking
import numpy as np
import RankingTree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
logging.basicConfig(level=logging.INFO)
class Scorer(object):
    def __init__(self, topics, pairs, mode):
        self.topics = topics
        self.pairs = pairs
        self.mode = mode
        self.persons = {}
        self.indicators = {}
        self.w2v = None
        self.eng = inflect.engine()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        for pair in self.pairs:
            if pair[0] not in self.persons:
                self.persons[pair[0]] = {mode:[pair[1]], 'text':""}
            else:
                self.persons[pair[0]][mode].append(pair[1])

    def score(self):
        pass

    def singularize(self, word):
        if self.eng:
            singular = self.eng.singular_noun(word)
            if singular:
                return singular
            else:
                return word

    def loadWord2Vec(self):
        if not self.w2v:
            try:
                self.w2v = Word2Vec.load('wiki.model')
            except Exception:
                self.logger.warn("message")
                return False
        return self.w2v

    def writeScore(self, out):
        self.logger.info("Writing at:%s", out)
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
                if len(p) == 3:
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
                pairs.append([l[0],l[1],l[2]])
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
        #session = mwapi.Session("https://en.wikipedia.org", user_agent = "test" )
        for p in persons:
            if not os.path.isfile('articles/'+p):
                try:
                    logging.info("Requesting %s", p)
                    page = Scorer.fetchArticleCirrus(p)
                    persons[p]['text'] = page['text']
                    persons[p]['opening_text'] = page['opening_text']
                    persons[p]['category'] = page['category']
                    f = open('articles/'+p, 'wb')
                    logging.info("Writing back: %s", p)
                    pickle.dump(page, f)
                    f.close()
                except Exception as e:
                    print("Failed for:"+p)
                    logging.error("Falied for: %s\n %s", p, str(e))
            else:
                logging.info("Reading local: %s", p)
                f = open('articles/'+p, 'rb')
                page = pickle.load(f)
                persons[p]['text'] = Scorer.process_sentence(page['text'])
                persons[p]['opening_text'] = Scorer.process_sentence(page['opening_text'])
                persons[p]['category'] = page['category']
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
    def fetchArticleCirrus(title):
        pagecontent = {}
        url = "http://en.wikipedia.org/wiki/"+title
        params = {'action': 'cirrusDump'}
        resp = requests.get(url=url, params=params)
        data = json.loads(resp.text)
        doc = data[0]['_source']
        pagecontent['text'] = doc["text"]
        pagecontent['opening_text'] = doc['opening_text']
        pagecontent['category'] = doc['category']
        return pagecontent

    @staticmethod
    def repl(m):
        return ''
    
    @staticmethod
    def process_sentence(sentence):
        # a heavenly regex, replaces mentions with actual names!
        #t = re.sub(r"\[(.*?)\|(.*?)\]",r"\1",sentence)
        t = re.sub(r"[\[\]%<>,\";]", " ", sentence, flags=re.UNICODE)
        #t = re.sub(r"http.*", " ",t)
        #t = re.sub(r"\-", " ", t)
        #t = re.sub(r"[^a-zA-Z_ \n]", " ", t, flags=re.UNICODE)
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

    def vec_avg(self, word):
        wordlist = word.lower().split()
        v = self.w2v[wordlist[0]]
        for w in wordlist[1:]:
            try:
                v = v + self.w2v[w]
            except Exception:
                self.logger.info("Average exception for:%s", w)
        return v / len(wordlist)

    def sanitize(self, word):
        # Same as for word2vec generation
        text=re.sub(r'[\[\]\(\)\:"\.;\',]',r'', word)
        return text.lower().replace(" ", "_")

    def getIndicators(self):
        if self.indicators:
            return self.indicators
        self.loadWord2Vec()
        self.indicators = {}
        topicwords = {}
        if self.mode == 'profession':
            topicwords = self.getTopicData('data/indicators-pro')
        else:
            topicwords = self.getTopicData('data/indicators-nat')
        for topic in self.topics:
            if topic not in topicwords:
                topicwords[topic] = []
                wordvec = self.vec_avg(topic.lower())
                words = self.w2v.similar_by_vector(wordvec)
                for w in words[0:10]:
                    topicwords[topic].append(w[0])
                logoutp = '[{}]'.format(topic)
                self.logger.info(logoutp+','.join(topicwords[topic]))
            topicwords[topic].extend(topic.lower().split())
            topicwords[topic] = list(set(topicwords[topic])) # remove duplicates
            topicwords[topic] = list(filter(None, topicwords[topic])) # remove empty
            
        self.indicators = topicwords
        return self.indicators

class CountComb(CountScorer):
    def __init__(self, topics, pairs, mode):
        super().__init__(topics, pairs, mode)

    def scoreLogistic(self, per, profession, article):
        if os.path.exists('data/'+profession+'.model'):
            # self.logger.info("Loading logistic model for %s", profession)
            self.logit = pickle.load(open('data/'+profession+'.model','rb'))
            vec = self.logit.get_params()['vect']
            logr = self.logit.get_params()['clf']
            X_test = vec.transform([article])
            label = logr.predict(X_test)
            #if label[0] == 1:
                #self.logger.info("Prediction 1 for %s,%s", per, profession)
            return label
        return 0
    
    def score(self):
        self.persons = Scorer.getWikipediaTexts2(self.persons)
        dummy = ""
        topicwords = self.getIndicators()
        self.loadWord2Vec()
        self.logger.info("Starting Scoring")
        i = 0
        for pair in self.pairs:
            per = pair[0]
            #print(dummy,",",per)
            prof = pair[1]
            if dummy != per:
                proflist = self.persons[per][self.mode]
                self.logger.info("Scoring %s", per)
                scorelist = self.scorePerson(per, self.persons[per]['text'], proflist, topicwords)
                scorelist = Scorer.maplin(scorelist)
                # assign the scores to persons pair
                for j in range(i, i + len(scorelist)):
                    #print("Scoring %s, %s:%d",self.pairs[j][0], self.pairs[j][1],scorelist[j-i])
                    logScore = self.scoreLogistic(per, self.pairs[j][1], self.persons[per]['text'])
                    finalScore = scorelist[j-i]
                    if logScore == 1:
                        finalscore = finalScore + finalScore / 2
                        if finalScore > 7:
                            finalScore = 7
                    self.pairs[j].append(finalScore)
            dummy = per
            i = i + 1

class SVMScorer(CountComb):
    """
    SVM Scorer with features:
    0:no. of profession occureces in first para
    1:no. of indicators in text
    2:no. of profession occurences in categories
    3:Average word vector cosine distance
    """
    def __init__(self, topics, pairs, mode):
        super().__init__(topics, pairs, mode)
        self.num_features = 6

    def count(self, per, article, words, normalize = True):
        text = article
        countscore = 0
        wc = 1
        if normalize:
            wc = len(article)
        wordlist = {self.singularize(w.lower()):0 for w in words}
        for w in text.split():
            tmp_word = self.singularize(w.lower())
            if tmp_word in wordlist:
                wordlist[tmp_word] = wordlist[tmp_word] + 1
        for k,v in wordlist.items():
            countscore = countscore + v
        return countscore / wc

    def vecAvg(self, per, words):
        per = self.sanitize(per)
        similarity  = 0
        count = 0
        if not self.loadWord2Vec():
            #logger.warn("Skipping word2vec")
            return similarity
        for w in words:
            try:
                sim = self.w2v.similarity(per, w)
                similarity = similarity + sim
                count = count + 1
            except Exception:
                self.logger.warn("Execption while similarity:%s,%s", per, w)
        count = count if count != 0 else 1
        return similarity / count

    def binaryCount(self, token, slist):
        token = self.singularize(token.lower())
        for s in slist:
            line = [self.singularize(t.lower()) for t in s.split()]
            if token in line:
                return True
        return False
    
    def feature0(self, per, header, entlist, topicwords):
        """
        Binary for now
        """
        # First three lines
        text = header.lower().split("\.")[0:10]
        features = []
        for p in entlist:
            present = False
            for topic in p.split():
                if self.binaryCount(topic, text):
                    present = True
            if present:
                features.append(1)
            else:
                features.append(0)
        return features
    
    def feature1(self, per, text, entlist, topicwords):
        features = []
        for p in entlist:
            try:
                features.append(self.count(per, text, topicwords[p]))
            except Exception:
                self.logger.exception("message")
                pdb.set_trace()
        # Normalize
        maxV = max(features)
        features = [f / maxV for f in features]
        return features

    def feature2(self, per, category, entlist, topicwords ):
        """
        Binary for now
        """
        features = []
        for p in entlist:
            present = False
            #for w in topicwords[p]:
            for tok in p.split():
                if self.binaryCount(tok, category):
                    present = True
            if not present:
                features.append(0)
            else:
                features.append(1)
        return features

    def feature3(self, per, entlist, topicwords):
        features = []
        for p in entlist:
            features.append(self.vecAvg(per, topicwords[p]))
        return features
    
    def feature4(self, per, text, entlist, topicwords):
        features = []
        for p in entlist:
            try:
                features.append(self.count(per, text, topicwords[p], False))
            except Exception:
                self.logger.exception("message")
                pdb.set_trace()
        # Normalize
        maxV = max(features)
        if maxV == 0:
            maxV = 1
        features = [f / maxV for f in features]
        return features

    def feature5(self, per, categories, entlist, topicwords):
        features = []
        for p in entlist:
            features.append(self.count(per, ' '.join(categories), topicwords[p], False))
        features = [f / len(categories) for f in features]
        return features
    
    def makeFeatures(self, flush = False):
        """
        Feature matrix:
        feature0: p1 p2 p3...
        feature1: p1 p2 p3...
        .
        .
        """
        if not flush and os.path.isfile('features.vec'):
            return self.readFeaturesFromFile('features.vec')
        
        self.persons = Scorer.getWikipediaTexts2(self.persons)
        dummy = ""
        topicwords = self.getIndicators()
        self.loadWord2Vec()
        self.logger.info("Starting feature generation")
        i = 0
        for pair in self.pairs:
            per = pair[0]
            #print(dummy,",",per)
            prof = pair[1]
            if dummy != per:
                entlist = self.persons[per][self.mode]
                self.logger.info("Features for %s", per)
                self.persons[per]['features'] = []
                self.persons[per]['features'].append(self.feature0(per, self.persons[per]['opening_text'], entlist, topicwords))
                self.persons[per]['features'].append(self.feature1(per, self.persons[per]['text'], entlist, topicwords))
                self.persons[per]['features'].append(self.feature2(per, self.persons[per]['category'], entlist, topicwords))
                self.persons[per]['features'].append(self.feature3(per, entlist, topicwords))
                self.persons[per]['features'].append(self.feature4(per, self.persons[per]['opening_text'], entlist, topicwords))
                self.persons[per]['features'].append(self.feature5(per, self.persons[per]['category'], entlist, topicwords))
                # write these features    
            dummy = per
            i = i + 1
        self.writeFeatures('features.vec')
        return self.genFeatures(toString = False)

    def multiclass(self):
        X,Y = self.makeFeatures()
        #pdb.set_trace()
        X = np.asarray(X,dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=0.75, random_state=0)
        ovr=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        print(ovr.score(X_test,y_test))
    
    def rf(self):
        X,Y = self.makeFeatures()
        X = np.asarray(X,dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        y = Y[:,0]
        group = Y[:,1]
        gkf = GroupKFold(n_splits=6)
        for train, test in gkf.split(X, y, groups=group):
            rfc = RankingTree.RankTree(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True, min_samples_split=0.15)
            rfc.fit(X[train], y[train])
            print(rfc.score(X[test], y[test]))

    def rfsearch(self):
        X,Y = self.makeFeatures()
        X = np.asarray(X,dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        y = Y[:,0]
        group = Y[:,1]
        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.75, random_state=0)
        rfc = RankingTree.RankTree(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
        tuned_parameters = [{'n_estimators':[10,100], 'max_features':['sqrt','log2'], 'min_samples_split':[0.05, 0.1, 0.15, 0.20]}]
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=tuned_parameters, cv= 5)
        CV_rfc.fit(X_train, y_train)
        means = CV_rfc.cv_results_['mean_test_score']
        stds = CV_rfc.cv_results_['std_test_score']
        print (CV_rfc.best_params_)
        for mean, std, params in zip(means, stds, CV_rfc.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print(CV_rfc.best_estimator_.feature_importances_)

    def rankSVM(self):
        X,Y = self.makeFeatures()
        X = np.asarray(X,dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        y = Y
        group = Y[:,1]
        gkf = GroupKFold(n_splits=10)
        for train, test in gkf.split(X, y, groups=group):
            svc = ranking.RankSVM(C=1e-1, kernel='rbf', gamma=10)
            svc.fit(X[train], y[train])
            print(svc.score(X[test], y[test]))
    
    def rank(self):
        X,Y = self.makeFeatures()
        #pdb.set_trace()
        X = np.asarray(X,dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=0.75, random_state=0)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1,1,10],
                                 'C': [1e-2,1e-1,1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        clf = GridSearchCV(ranking.RankSVM(C=1), tuned_parameters, cv=5)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        be = clf.best_estimator_
        #print("Detailed classification report:")
        #pdb.set_trace()
        #y_true, y_pred = y_test[0], clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        #kf = KFold(n_splits=5)
        #for train, test in kf.split(X):
        # make a simple plot out of it
        #import pylab as pl
        #pl.scatter(np.dot(X, true_coef), y)
        #pl.title('Data to be learned')
        #pl.xlabel('<X, coef>')
        #pl.ylabel('y')
        #pl.show()

        # print the performance of ranking
        #print(train) 
        #rank_svm = ranking.RankSVM().fit(X[train], Y[train])
        #rank_svm = ranking.RankSVM()
        #scores = cross_val_score(rank_svm, X, y, cv=5)
        #print ('Performance of ranking ', rank_svm.score(X[test], Y[test]))

    def featureSelect(self):
        X,y = self.makeFeatures()
        X = np.asarray(X,dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)[:,0]
        svc = ranking.RankSVM(kernel='linear', C=10)
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2))
        rfecv.fit(X, y)
        print("Optimal number of features : %d" % rfecv.n_features_)

    def writeFeatures(self, filename):
        outp = open(filename, 'w')
        outp.write(self.genFeatures())
        outp.close()

    def readFeaturesFromFile(self, filename):
        self.logger.info("Reading features locally from %s", filename)
        X = []
        y = []
        f = open(filename, 'r')
        for line in f:
            data = line.strip()
            if data.startswith( '#' ):
                continue
            features = data.split()
            y.append([features[0], features[1].split(':')[1]])
            X_l = []
            for x in features[2:]:
                if ':' in x:
                    X_l.append(x.split(':')[1])
            X.append(X_l)
        f.close()
        return X,y

    def genFeatures(self, toString = True):
        featureString = ""
        featureVec = []
        X = []
        y = []
        line = "{} qid:{}"
        for i in range(0, self.num_features):
            line = line + " " + str(i+1) + ":{}"
        line = line + " # {}" + "\n"
        i = 1 # person-index
        j = 0 # pair-index
        dummy = ""
        for pair in self.pairs:
            per = pair[0]
            #print(dummy,",",per)
            if dummy != per:
                features = self.persons[per]['features']
                # Iterate over professions by scanning any row
                for p in range(0, len(features[0])):
                    args = []
                    featurelist = []
                    for f in range(0, len(features)):
                        featurelist.append(features[f][p])
                    if toString:
                        args.append(self.pairs[j][2])
                        args.append(i)
                        args.extend(featurelist)
                        args.append(per)
                        formatted = line.format(*args)
                        featureString = featureString + formatted
                    else:
                        y.append([self.pairs[j][2], i])
                        X.append(featurelist)
                    j = j + 1
                featureString = featureString + "#\n"
                i = i + 1
            dummy = per
        if toString:
            return featureString
        else:
            return X,y

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

    logging.info("Getting topics")
    topics = Scorer.getTopics(input+'/professions')
    logging.info("Getting pairs")
    pairs = Scorer.getPersonPairs(input+'/profession.train')
    sc = SVMScorer(topics, pairs, 'profession')
    #sc.makeFeatures(flush=True)
    #sc.writeScore(output+'/profession.out')
    #sc.rank()
    #sc.featureSelect()
    #sc.writeFeatures()
    #sc.rf()
    sc.rankSVM()
if __name__ == '__main__':
    main()
