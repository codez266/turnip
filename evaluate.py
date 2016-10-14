from gensim.models import Word2Vec
import math
import sys

def getSimilarity(name, profession, model):
    try:
        return model.similarity(name,profession)
    except KeyError as e:
        #print("Key error for: "+name)
        return 0


def maplog(list):
    max = -1
    mapped = []
    for s in list:
        if max < s:
            max = s
    for i,s in enumerate(list):
        if max != 0:
            if s > 0:
                mapped.append( s / max )
                mapped[i] = mapped[i] * ( 2**7 )
                mapped[i] = int(math.log(mapped[i],2))
            else:
                mapped.append( 0 )
        else:
            mapped.append(s)
    return mapped


def maplin(list):
    max = -1
    mapped = []
    for s in list:
        if max < s:
            max = s
    for i, s in enumerate(list):
        if max != 0:
            if s > 0:
                mapped.append( s / max )
                mapped[i] = int(mapped[i] * (7))
            else:
                mapped.append(0)
        else:
            mapped.append(s)
    return mapped


def compareProfession(person, professions, model):
    similarlist = []
    for profession in professions:
        per = person.replace(" ", "_")
        profwords = profession.split()
        sim = 0
        for p in profwords:
            similarity = getSimilarity(per.lower(), p, model)
            if sim < similarity:
                sim = similarity
        similarlist.append(sim)
    similarlist = maplin(similarlist)
    return similarlist


def listSimilar(train, model):
    model = Word2Vec.load(model)
    with open(train) as file:
        list = []
        prev = None
        for line in file:
            data = line.split('\t')
            person = data[0]
            if person != prev and prev is not None:
                out = ''
                similarlist = compareProfession(person.lower(), list, model)
                for i, s in enumerate(similarlist):
                    out = out + prev + "\t" + list[i] + '\t' + str(s) + '\n'
                print(out.strip("\n"))
                list = [data[1].lower()]
            else:
                list.append(data[1].lower())
            prev = person
        file.close()

if __name__ == '__main__':
    train = sys.argv[1]
    model = sys.argv[2]
    listSimilar(train, model)
