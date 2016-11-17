from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
import pickle
import random
import os
import math

from pprint import pprint
from time import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
class Vectorizer(object):
    def __init__(self, vec_file, data = 'turnip/data/'):
        self.file = vec_file
        self.docs = []
        self.data_dir = data
        self.perprofession = None
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

    def computeStatistics(self):
        self.vocab = self.vector.get_feature_names()
        self.terms = list(zip(self.vector.idf_, range(len(self.vector.idf_))))
        self.sortedterms = sorted(self.terms, key=lambda t:t[0], reverse=True)
    
    def readPersonProfessions(self):
        if not self.perprofession:
            self.perprofession = {}
            with open(self.data_dir+'profession.kb', 'rt') as file:
                for line in file:
                    text = line.split('\t')
                    p = text[0]
                    if p not in self.perprofession:
                        self.perprofession[p] = []
                    self.perprofession[p].append(text[1].strip())
        return self.perprofession

    def getPositivePersons(self, profession):
        if not self.perprofession:
            return False
        sample = []
        for per, profs in self.perprofession.items():
            if len(profs) == 1 and profs[0] == profession:
                sample.append(per)
        return sample

    def getNegativePersons(self, profession, n = -1):
        sample = []
        if not self.perprofession:
            return False
        persons = list(self.perprofession.keys())
        random.shuffle(persons)
        for p in persons:
            professions = self.perprofession[p]
            if profession not in professions:
                sample.append(p)
                # no more than n samples
                if n!=-1 and len(sample) > n:
                    break
        return sample

    def getSamples(self, profession):
        samples = []
        self.profession = profession
        self.readPersonProfessions()
        pos = self.getPositivePersons(profession)
        neg = self.getNegativePersons(profession, len(pos))
        return [pos, neg]

    def getText(self, filter_list):
        if os.path.isfile(self.data_dir+self.profession+'.docs') and filter_list != None:
            self.docs = pickle.load(open(self.data_dir+self.profession+'.docs','rb'))
            return self.docs
        docs = []
        posdocs = []
        negdocs = []
        with open('persons3') as file:
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
            with open(self.data_dir+self.profession+'.docs','wb') as f:
                pickle.dump(docs, f)
        
        return docs    

    def search(self, profession):
        
        self.readPersonProfessions()
        filter_list = self.getSamples(profession)
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
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()

        #for param_name in sorted(self.parameters.keys()):
        #    print("\t%s: %r" % (param_name, best_parameters[param_name])) 
        return [self.parameters, best_parameters]


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
    
    @staticmethod
    def indicators(ind_file, write_score):
        with open('params') as f:
            fwr = open(ind_file, 'w')
            allvec = pickle.load(open('turnip/data/all.vec','rb'))
            for line in f:
                text=line.strip()
                data = text.split('\t')
                if data[0] == "skipping" or len(data) < 2:
                    continue
                params = data[1].split(',')
                C = float(params[0])
                n_iters = float(params[1])
                solver = params[2]
                max_features = int(params[3])
                max_df = float(params[4])
                classer = Vectorizer('log')
                print("processing "+data[0])
                scores=classer.evaluate(data[0], C, n_iters, solver, max_features, max_df, allvec.idf_, allvec.vocabulary_)
                terms = scores[0]
                weights = scores[1]
                st = data[0] + "\t" + str(classer.clf.score(classer.X, classer.y)) + "\t"
                for t in (weights[0:10]):
                    if write_score:
                        st = st + terms[t[1]] + "(" + str(t[0]) + ")" + ","
                    else:
                        st = st + terms[t[1]] + ","
                st = st.strip(",")
                fwr.write(st+"\n")
            fwr.close()
if __name__ == "__main__":
    #i = 0
    #with open('turnip/data/professions') as f:
    #    fwr = open('params', 'a')
    #    fwr.write("max_df,max_features,C,solver,max_iter\n")
    #    for line in f:
    #        i = i + 1
    #        if i <= 4:
    #            continue
    #        profession=line.strip()
    #        vec=Vectorizer(profession)
    #        params = vec.search(profession)
    #        if params is None:
    #            fwr.write("skipping "+profession+"\n")
    #            continue
    #        st = profession + "\t"
    #        for k in params[0].keys():
    #            st = st + str(params[1][k]) + ","
    #        print(st)
    #        st = st.strip(",")
    #        fwr.write(st+"\n")
    #    fwr.close()
    Vectorizer.indicators("indicators-f", False)
    #with open('turnip/data/professions') as f:
    #    for line in f:
    #        profession = line.strip()
    #        vect = Vectorizer('v')
    #        vect.readPersonProfessions()
    #        l = vect.getPositivePersons(profession)
    #        if len(l) < 50:
    #            print (profession+"\t"+str(len(l)))
