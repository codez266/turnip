#!/usr/bin/python
# A python script for fetching wikidata ids for professions or nationalities
# using sparql endpoint
from SPARQLWrapper import SPARQLWrapper, JSON
import pdb
sparql = SPARQLWrapper('https://query.wikidata.org/bigdata/namespace/wdq/sparql')
pids = {}
with open('data/professions.ids') as fin:
	for l in fin:
		s = l.strip().split('\t')
		if len(s) == 3:
			pids[s[0]] = [s[1], s[2]]
with open('data/professions') as fin:
	fout = open('data/professions.ids','w')
	for l in fin:
		pr = l.strip()	
		if pr in pids:
			fout.write(pr + '\t' + pids[pr][0] + '\t' + pids[pr][1] + '\n')
			continue
		sparql.setQuery("""
			SELECT ?item WHERE{
				?item rdfs:label \"%s\"@en.
				?item wdt:P31 wd:Q28640.
			}""" % (pr.lower()))
		idval = ''
		fbid = ''
		try:
			sparql.setReturnFormat(JSON)
			results = sparql.query().convert()
			res = results['results']['bindings'][0]
			idval = res['item']['value']
			#fbid = res['fbid']['value']
		except Exception as e:
			print(e)
		fout.write(pr + '\t' + idval + '\t' + fbid + '\n')
	fout.close()

