from gensim.models import Word2Vec
import math
import argparse

DATA_DIR = "data/"


def getSimilarity(name, profession, model):
    try:
        return model.similarity(name, profession)
    except KeyError as e:
        # print("Key error for: "+name)
        return 0


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


def score(train, model, output, mode):
    model = Word2Vec.load(model)
    outp = open(output + "/" + train.rsplit("/", 1)[1], 'w')
    with open(train) as file:
        list = []
        prev = None
        for line in file:
            data = line.split('\t')
            person = data[0]
            if person != prev and prev is not None:
                out = ''
                similarlist = []
                # 0 for profession, 1 for nationality
                if mode == 0:
                    similarlist = compareProfession(person.lower(), list, model)
                # TODO nationality
                for i, s in enumerate(similarlist):
                    out = out + prev + "\t" + list[i].strip() + '\t' + str(s) + '\n'
                outp.write(out)
                list = [data[1].lower().strip()]
            else:
                list.append(data[1].lower().strip())
            prev = person
        file.close()
        outp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input directory", action="store_true")
    parser.add_argument("input", help="Input directory value")

    parser.add_argument("-o", help="Output directory", action="store_true")
    parser.add_argument("output", help="Output directory value")
    args = parser.parse_args()

    model = 'wiki-vectors.model'

    input = ""
    output = ""
    if args.i:
        input = args.input
    if args.o:
        output = args.output

    profession = []
    nationality = []
    with open(DATA_DIR+'professions') as file:
        for line in file:
            profession.append(line.strip())
        file.close()

    with open(DATA_DIR+'nationalities') as file:
        for line in file:
            nationality.append(line.strip())
        file.close()

    isProfession = True
    with open(input) as file:
        for line in file:
            data = line.split('\t')
            entity = data[1].strip()
            if entity in nationality:
                isProfession = False
            break
        file.close()
    if isProfession:
        # call profession
        score(input, model, output, 0)
    else:
        # call nationality
        score(input, model, output, 1)


if __name__ == '__main__':
    main()
