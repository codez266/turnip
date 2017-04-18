import urllib, os, json, requests, pdb, gzip, re
def fetchArticleSize(title):
	pagecontent = {}
	url = "http://en.wikipedia.org/wiki/"+title
	params = {'action': 'cirrusDump'}
	doc = {}
	try:
		resp = requests.get(url=url, params=params)
		data = json.loads(resp.text)
		doc = data[0]['_source']
	except Exception as e:
		print(title,e)
		return 0
	pagecontent['text'] = doc["text"]
	pagecontent['opening_text'] = doc['opening_text']
	pagecontent['category'] = doc['category']
	return len(pagecontent['text'])

names = []
with open('data/profession.one') as fin:
	#fout = open('data/profession.one.ws', 'w')
	for line in fin:
		name = line.strip().split('\t')[0]
		names.append(name)
		#sz = fetchArticleSize(name)
		#fout.write(line.strip() + '\t' + str(sz) + '\n')
	#fout.close()

index = 0
fout = open('data/profession.one.texts', 'w')
for line in gzip.open('../corpus/enwiki-20170403-cirrussearch-content.json.gz', 'r'):
	index = index + 1
	if index % 2 == 1:
		continue
	textline = json.loads(line.decode('utf-8'))
	try:
		if textline['title'] in names and textline['opening_text']:
			texts = textline['opening_text'].lower()
			#text=re.sub(r'\[(.*?)\|(.*?)\]',r'\1',textline)
			text=re.sub(r'[\[\]\:"\.;,]',r'', texts, re.UNICODE)
			text=re.sub(r'\s+',r' ', text, re.UNICODE)
			text=re.sub(r'[0-9]*', r' ', text, re.UNICODE)
			cat = textline['category']
			catstr = ''
			if cat:
				catstr = ' '.join(cat).lower()
			output = textline['title'] + '\t' + str(len(text)) + '\t' + text + '\t' + catstr + '\n'
			fout.write(output)
	except Exception as e:
		print(e, textline['title'])

fout.close()

