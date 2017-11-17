import pandas as pd
import numpy as np
import re

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag

def clean(sentences):
	"""
	clean() clean sentences of special characters,
	removes stopwords, and stems.
	:param sentences: list or series of strings
	:return: list of cleaned strings
	"""

	## initialize stemmers and stop words
	stemmer = PorterStemmer()
	stop = list(set(stopwords.words('english')))
    
	## tokenize
	corp = [word_tokenize(re.sub("[^a-z\s]", "", s.lower())) for s in sentences]
    
	## remove stopwords (first sweep)
	corp = [[w for w in sen if w not in stop] for sen in corp]
    
	## stem words
	corp = [[stemmer.stem(w) for w in sen] for sen in corp]

	## remove stopwords (second sweep)
	corp = [[w for w in sen if w not in stop] for sen in corp]

	## concatenate tokens into strings and return as list of strings
	return [" ".join(c) for c in corp]

def characterCount(items):
	"""
	characterCount() list of length values
	:param items: list of strings
	:returns: list of integers
	"""
	return [len(i) for i in items]

def posCount(items):
	"""
	posCount() transforms pos tag
	results for a string and returns
	as a structured dataframe of pos counts
	:param items: list of strings
	:returns: pandas df 
	"""

	# initialize empty df to store results
	results = pd.DataFrame()

	# iterate through strings
	for i in items:
		# for each string initialize dict to store pos results
		row = {}
		# for each string, iterate through pos tags for each token
		for t in pos_tag(i):
			# update count for tag's dict key
			if row.get(t[1], None) is not None:
				row[t[1]] += 1
			else:
				row[t[1]] = 1
		# transform dictionary to series and append to df
		results = results.append(pd.DataFrame(pd.Series(row)).T)
	return results.fillna(0).reset_index(drop=True)

def findUnique(items, pop):
	"""
	findUnique() finds elements in items
	that are unique to items when cross-referenced against pop
	:param items: list we want to find unique elements
	:param pop: list we're cross-referencing items against
	:returns: list of unique elements
	"""
	return list(set([i for i in items if i not in pop]))

def tagUnique(text, uniq_tok):
	u_df = pd.DataFrame()
	for t in text:
		matches = []
		for e in uniq_tok:
			if len(set(word_tokenize(t)).intersection(e)) > 0:
				matches.append(1)
			else:
				matches.append(0)
		u_df = u_df.append(pd.DataFrame(matches).T)
	return u_df.reset_index(drop=True)

def itemCount(elements, item):
	return [len([i for i in e.split(" ") if i == item]) for e in elements]

def alliteration(text):
	result = []
	for t in text:
		score = 0
		tokens = word_tokenize(t)
		for n in range(len(tokens)-1):
			if tokens[n][0] == tokens[n+1][0]:
				score += 1
		result.append(score)
	return result

def assonance(text):
	result = []
	for t in text:
		score = 0
		tokens = word_tokenize(t)
		tokens = [re.sub("[^aeiou]", "", t) for t in tokens]
		for n in range(len(tokens)-1):
			if len(tokens[n]) > 0 and len(tokens[n+1]) > 0:
				if tokens[n][0] == tokens[n+1][0]:
					score += 1
		result.append(score)
	return result

def vowelRatio(text):
	return [len(re.sub("[^aeiouy]", "", t)) / float(len(t)+1) for t in text]

def conjunction(text):
	words = [
		"and",
		"or",
		"but",
		"nor",
		"so",
		"for",
		"yet",
		"after",
		"although",
		"as",
		"because",
		"before",
		"once",
		"since",
		"though",
		"till",
		"unless",
		"until",
		"what",
		"when",
		"wherever",
		"whether",
		"while"
	]
	return [len(set([i for i in t.lower().split(" ") if i in words])) for t in text]

def repetition(text):
	results = []
	for t in text:
		score = 0
		tokens = word_tokenize(t)
		for n in range(len(tokens)-1):
			if tokens[n] == tokens[n+1]:
				score += 1
		results.append(score)
	return results

def stopCount(text):
	stop = list(set(stopwords.words('english')))
	return [len([i for i in t.lower().split(" ") if i in stop]) for t in text]

def startWords(text):
	return [" ".join(re.findall("(^[A-Z][A-Za-z]*|\. [A-Z][A-Za-z]*)", t)) for t in text]

def padSingle(text):
	return [t + "xxxx" if len(t) == 1 else t for t in text]