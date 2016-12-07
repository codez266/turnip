from scorer import *
import pdb
def scoreExternal():
    logging.info("Getting topics")
    input = 'data'
    topics = Scorer.getTopics(input+'/professions')
    logging.info("Getting pairs")
    pairs = Scorer.getPersonPairs(input+'/profession.train')
    sc = SVMScorer(topics, pairs, 'profession')
    f = open('svm1/predictions', 'r')
    fo = open('profession.out', 'w')
    i = 0
    j = 0
    score = []
    entity = 'profession'
    dummy = ""
    for line in f:
        score.append(float(line.strip()))

    for pair in sc.pairs:
        per = pair[0]
        if dummy != per:
            scores = []
            entities = []
            try:
                for ent in sc.persons[per][entity]:
                    scores.append(score[j])
                    entities.append((per,ent))
                    j = j+1
            except Exception:
                pdb.set_trace()
            scores = sc.maplin(scores)
            for s in zip(entities, scores):
                print(s)
                line = "{}\t{}\t{}\n".format(s[0][0], s[0][1], s[1])
                fo.write(line)
        dummy = per
        i = i + 1

if __name__ == "__main__":
    scoreExternal()
