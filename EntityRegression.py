from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import logging
import pickle
import random
import os
import math
import numpy as np

from pprint import pprint
from time import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
class Vectorizer(object):
    def __init__(self, vec_file, switch = 0, data = 'data/'):
        """
        switch:0-profession, nationality otherwise
        """
        self.switch = switch
        self.file = vec_file
        self.docs = []
        self.data_dir = data
        self.perprofession = None
        self.pernational = None
        self.pipeline = Pipeline([
                ('vect', TfidfVectorizer(use_idf=False, norm='l1',stop_words='english')),
                ('clf', LogisticRegression())
            ])
        self.parameters = {
                'vect__max_df': (0.8,1.0),
                'vect__max_features': (1000,2000,5000),
                'clf__C': (0.1,1,10),
                'clf__solver': ('liblinear', 'sag'),
                'clf__max_iter': (1e5, 1e6)
                }
    def getTerms(self):
        list = []
        for t in self.terms:
            index = t[1]
            word = self.vocab[index]
            try:
                val=int(word)
            except ValueError:
                list.append((str(word), t[0]))
        return list

    def loadVector(self):
        self.vector = pickle.load(open(self.file,'rb'))
        self.computeStatistics()
    
    def readPersonEntities(self):
        if self.switch == 0 and self.perprofession:
            return self.perprofession
        if self.switch == 1 and self.pernational:
            return self.pernational
        db = {}
        dbfile = 'profession.kb' if self.switch == 0 else 'nationality.kb'
        with open(self.data_dir+dbfile, 'rt') as file:
            for line in file:
                text = line.split('\t')
                p = text[0]
                if p not in db:
                    db[p] = []
                db[p].append(text[1].strip())
        if self.switch == 0:
            self.perprofession = db
        else:
            self.pernational = db
        return db

    def getPositivePersons(self, entity):
        if self.switch == 0 and not self.perprofession:
            return False
        if self.switch == 1 and not self.pernational:
            return False
        sample = []
        db = self.perprofession if self.switch == 0 else self.pernational
        for per, vals in db.items():
            if len(vals) == 1 and vals[0] == entity:
                sample.append(per)
        return sample

    def getNegativePersons(self, entity, n = -1):
        sample = []
        if self.switch == 0 and not self.perprofession:
            return False
        if self.switch == 1 and not self.pernational:
            return False
        db = self.perprofession if self.switch == 0 else self.pernational
        persons = list(db.keys())
        random.shuffle(persons)
        for p in persons:
            entities = db[p]
            if entity not in entities:
                sample.append(p)
                # no more than n samples
                if n!=-1 and len(sample) > n:
                    break
        return sample

    def getSamples(self, entity):
        samples = []
        self.entity = entity
        self.readPersonEntities()
        pos = self.getPositivePersons(entity)
        neg = self.getNegativePersons(entity, len(pos))
        return [pos, neg]

    def getText(self, filter_list):
        if os.path.isfile(self.data_dir+self.entity+'.docs') and filter_list != None:
            self.docs = pickle.load(open(self.data_dir+self.entity+'.docs','rb'))
            return self.docs
        docs = []
        posdocs = []
        negdocs = []
        with open('../persons3') as file:
            for line in file:
                text = line.split('\t')
                if not filter_list:
                    docs.append(text[1].lower())
                elif text[0] in filter_list[0]:
                    posdocs.append(text[1].lower())
                elif text[0] in filter_list[1]:
                    negdocs.append(text[1].lower())
        if filter_list != None:
            docs = [posdocs+negdocs, [1]*len(posdocs)+[0]*len(negdocs)]
            #with open(self.data_dir+self.profession+'.docs','wb') as f:
            #    pickle.dump(docs, f)
        
        return docs    

    def TfScorer(self, entity, maxf = 2000):
        self.readPersonEntities()
        filter_list = self.getSamples(entity)
        docs = self.getText(filter_list)
        ones = docs[1].count(1)
        text = docs[0][0:ones]
        if len(text) < 3:
            return None
        self.vec = TfidfVectorizer(stop_words='english',norm='l2',use_idf=True,max_features = maxf)
        logger.info("Fitting %s", profession)
        X = self.vec.fit_transform(text).toarray()
        scores = np.mean(X, axis=0)
        scorelist = list(zip(scores, range(len(scores))))
        scorelist = sorted(scorelist, key=lambda t:t[0], reverse = True)
        return [self.vec, scorelist]
    
    @staticmethod
    def evaluateProfession():
        data_dir = 'data/'
        with open(data_dir + 'professions', 'r') as f:
            fout = open(data_dir + 'indicators-pro', 'w')
            skipped = []
            for line in f:
                l = line.strip()
                v = Vectorizer('', 0)
                logger.info("Evaluatin %s", l)
                tf = v.TfScorer(l)
                if tf is None:
                    skipped.append(l)
                else:
                    weights = tf[1]
                    terms = tf[0].get_feature_names()
                    proline = l + '\t'
                    for idx, i in enumerate(weights[0:15]):
                        term = terms[i[1]]
                        proline = proline + term + ','
                    proline.strip(',')
                    proline = proline + '\n'
                    fout.write(proline)
            fout.write('skipped\t')
            skiplist = ''
            for s in skipped:
                skiplist = skiplist + s + ','
            skiplist.strip(',')
            fout.write(skiplist+'\n')

    def gridsearch(self, entity):
        if os.path.isfile(self.data_dir + entity + '.model'): 
            logger.info("Fetching %s", entity)
            self.clf = pickle.load(open(self.data_dir+entity+'.model','rb'))
            return [self.clf, -1]
        self.readPersonEntities()
        filter_list = self.getSamples(entity)
        docs = self.getText(filter_list)
        if len(docs[0])<=3:
            return None
        #print(len(docs[0]))
        #print(len(docs[1]))
        grid_search = GridSearchCV(self.pipeline, self.parameters, n_jobs=-1, verbose=1)
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in self.pipeline.steps])
        print("parameters:")
        pprint(self.parameters)
        t0 = time()
        grid_search.fit(docs[0], docs[1])
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        self.clf = grid_search.best_estimator_
        # Return nothing is score is less than 70%
        if grid_search.best_score_ <= 0.7:
            return None
        logger.info("Saving %s", entity)
        with open(self.data_dir+entity+'.model','wb') as f:
            pickle.dump(self.clf, f)
        return [self.clf, grid_search.best_score_]

    @staticmethod
    def evaluateNationality():
        data_dir = 'data/'
        with open(data_dir + 'nationalities', 'r') as f:
            fout = open(data_dir + 'indicators-nat', 'a')
            skipped = []
            index = -1
            for line in f:
                l = line.strip()
                index = index + 1
                if l.lower() != "united states of america" and index >= 66:
                    v = Vectorizer('', 1)
                    gs = v.gridsearch(l)
                    if gs is None or gs[1] == -1:
                        skipped.append(l)
                    else:
                        clf = gs[0]
                        vec = clf.get_params()['vect']
                        logr = clf.get_params()['clf']
                        weights = list(zip(logr.coef_[0], range(len(logr.coef_[0]))))
                        terms = vec.get_feature_names()
                        weights = sorted(weights, key = lambda t:t[0], reverse = True)
                        best_score = 'N/A'
                        natline = l
                        if gs[1] != -1:
                            best_score = str(gs[1])
                        natline = natline + '\t' + best_score + '\t'
                        for idx, i in enumerate(weights[0:15]):
                            term = terms[i[1]]
                            natline = natline + term + ','
                        natline.strip(',')
                        natline = natline + '\n'
                        fout.write(natline)
            fout.write('skipped\t')
            skiplist = ''
            for s in skipped:
                skiplist = skiplist + s + ','
            skiplist.strip(',')
            fout.write(skiplist+'\n')

    @staticmethod
    def evaluateProfessionReg():
        data_dir = 'data/'
        with open(data_dir + 'professions', 'r') as f:
            fout = open(data_dir + 'indicators-pro', 'a')
            skipped = []
            for line in f:
                l = line.strip()
                v = Vectorizer('', 0)
                gs = v.gridsearch(l)
                if gs is None or gs[1] == -1:
                    skipped.append(l)
                else:
                    clf = gs[0]
                    vec = clf.get_params()['vect']
                    logr = clf.get_params()['clf']
                    weights = list(zip(logr.coef_[0], range(len(logr.coef_[0]))))
                    terms = vec.get_feature_names()
                    weights = sorted(weights, key = lambda t:t[0], reverse = True)
                    best_score = 'N/A'
                    if gs[1] != -1:
                        best_score = str(gs[1])
                    proline = l
                    proline = proline + '\t' + best_score + '\t'
                    for idx, i in enumerate(weights[0:15]):
                        term = terms[i[1]]
                        proline = proline + term + ','
                    proline.strip(',')
                    proline = proline + '\n'
                    fout.write(proline)
            fout.write('skipped\t')
            skiplist = ''
            for s in skipped:
                skiplist = skiplist + s + ','
            skiplist.strip(',')
            fout.write(skiplist+'\n')

    def evaluate(self, profession, C, max_iter, solver, max_features, max_df, idf, idf_voc):
        self.readPersonProfessions()
        filter_list = self.getSamples(profession)
        docs = self.getText(filter_list)
        self.vec = TfidfVectorizer(stop_words='english',norm='l1',use_idf=False,max_features = max_features)
        positive = docs[1].count(1)
        posdocs = docs[0][0:positive]
        #self.posvec = TfidfVectorizer(stop_words='english', norm='l1', max_features = max_features)
        #posMat = self.posvec.fit_transform(posdocs).toarray()
        #self.tf = posMat.sum(axis=0) / len(posdocs)
        mat=self.vec.fit_transform(docs[0])
        self.X=mat.toarray()
        self.y=docs[1]
        self.clf = LogisticRegression(C=C, max_iter = max_iter, solver = solver)
        self.clf.fit(self.X,self.y)
        weights = list(zip(self.clf.coef_[0],range(len(self.clf.coef_[0]))))
        terms = self.vec.get_feature_names()
        norm_weights = []
        for idx,i in enumerate(weights):
            term = terms[i[1]]
            if term in idf_voc:
                norm_weights.append(( i[0] * idf[idf_voc[term]], i[1]))
            else:
                norm_weights.append(( 0, i[1]))
        norm_weights = sorted(norm_weights, key=lambda t:t[0], reverse=True)
        return [terms, norm_weights]

    def rfClf(self, entity, maxf = 5000):
        self.readPersonEntities()
        filter_list = self.getSamples(entity)
        docs = self.getText(filter_list)
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words='english', norm='l1', use_idf=True, max_features=maxf)
        positive = docs[1].count(1)
        posdocs = docs[0][0:positive]
        self.X = self.vec.fit_transform(docs[0])
        self.y = np.asarray(docs[1])
        kf = KFold(n_splits=5)
        for train, test in kf.split(self.y):
            self.clf = RandomForestClassifier(n_estimators=10)
            self.clf.fit(self.X[train], self.y[train])
            print(self.clf.score(self.X[test], self.y[test]))
    
#if __name__ == "__main__":
    #Vectorizer.evaluateNationality()
