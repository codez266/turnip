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
    def maplog(list):
        max = -1
        mapped = []
        for s in list:
            if max < s:
                max = s
        for i, s in enumerate(list):
            if max != 0:
                if s > 0:
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
        self.persons = Scorer.getWikipediaTexts('../persons3', self.persons)
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
                        words = [x.lower() for x in topicwords[p]]
                        #print(words)
                        #print("calculating for:", per, p)
                        text = self.persons[per]['text']
                        score = 0
                        for w in text.split():
                            if w in words:
                                score = score + 1
                        #print(score)
                        scorelist.append(score)
                    else:
                        scorelist.append(0)
                # all professions scored
                scorelist = Scorer.maplin(scorelist)
                # assign the scores to persons pair
                for j in range(i, i + len(scorelist)):
                    #print("Scoring %s, %s:%d",self.pairs[j][0], self.pairs[j][1],scorelist[j-i])
                    self.pairs[j].append(scorelist[j-i])
            dummy = per
            i = i + 1
        self.writeScore('data/profession.out')

topics = Scorer.getTopics('data/professions')
pairs = Scorer.getPersonPairs('data/profession.train')
sc = CountScorer(topics, pairs, 'profession')
sc.score()

