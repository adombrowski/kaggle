import pandas as pd
import numpy as np
import pickle
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag

def clean(sentences):
	## initialize stemmers and stop words
	stemmer = PorterStemmer()
	stop = list(set(stopwords.words('english')))
    
	## tokenize
	corp = [word_tokenize(re.sub("[^a-z]", " ", s.lower())) for s in sentences]
    
	## remove stopwords (first sweep)
	corp = [[w for w in sen if w not in stop] for sen in corp]
    
	## stem words
	corp = [[stemmer.stem(w) for w in sen] for sen in corp]

	## remove stopwords (second sweep)
	corp = [[w for w in sen if w not in stop] for sen in corp]

	return [" ".join(c) for c in corp]

def characterCount(items):
	return [len(list(i)) for i in items]

def posCount(items):
	results = pd.DataFrame()
	for i in items:
		row = {}
		for t in pos_tag(i):
			if row.get(t[1], None) is not None:
				row[t[1]] += 1
			else:
				row[t[1]] = 1
		row = pd.DataFrame(pd.Series(row)).T
		results = results.append(row)
	return results.fillna(0).reset_index(drop=True)

def findUnique(items, pop):
    return list(set([i for i in items if i not in pop]))

def tagUnique(train):
	## set unique
	eap = [item for sublist in [word_tokenize(c) for c in train.clean_text[train.author=="EAP"]] for item in sublist]
	hpl = [item for sublist in [word_tokenize(c) for c in train.clean_text[train.author=="HPL"]] for item in sublist]
	mws = [item for sublist in [word_tokenize(c) for c in train.clean_text[train.author=="MWS"]] for item in sublist]

	uniq_tok = [
		set(findUnique(eap, set(hpl).union(set(mws)))),
		set(findUnique(hpl, set(eap).union(set(mws)))),
		set(findUnique(mws, set(hpl).union(set(eap))))
	]
	u_df = pd.DataFrame()
	for t in train.clean_text:
		matches = []
		for e in uniq_tok:
			if len(set(word_tokenize(t)).intersection(e)) > 0:
				matches.append(1)
			else:
				matches.append(0)
		u_df = u_df.append(pd.DataFrame(matches).T)
	return u_df.reset_index(drop=True)

def main():
	## import train.csv
	train = pd.read_csv("train.csv", sep=",")

	## clean text samples
	print("Cleaning...")
	train['clean_text'] = clean(train.text)
	print("Clean complete...")

	## split data in half
	train_base = train.sample(frac=.5, random_state = 0)
	train_stack = train[~train.id.isin(train_base.id)]
	train_base, train_stack = train_base.reset_index(drop=True), train_stack.reset_index(drop=True)

	"""
	Step 1:
		Train base tfidf classifiers, mnb
	"""

	## initialize tfidf vectorizer and transform data
	print("Training MNB...")
	tfidf_vectorizer = TfidfVectorizer(max_features=4000).fit(train_base.clean_text)
	pickle.dump(tfidf_vectorizer, open("vectorizer.pickle", "wb"))

	tfidf_base = tfidf_vectorizer.transform(train_base.clean_text)
	tfidf_stack = tfidf_vectorizer.transform(train_stack.clean_text)

	## set targets
	y = train_base.author.as_matrix().ravel()

	## train base classifiers
	mnb = MultinomialNB(alpha=.1).fit(tfidf_base, y)

	"""
	Step 2:
		Prepare data for stack classifier
	"""

	## generate pos data
	print("Calculating POS...")
	pos = posCount(train_stack.text)
	print("Running SVD on POS data...")
	pos = TruncatedSVD(n_components=2).fit_transform(pos.as_matrix())

	## store in df
	meta = pd.concat([pd.DataFrame(pos), pd.DataFrame(mnb.predict_proba(tfidf_stack))], axis=1)

	## calculate char count
	meta['char_count'] = characterCount(train_stack.clean_text)

	## tag unique
	meta = pd.concat([meta, tagUnique(train_stack)], axis=1)

	## scale data
	print("Scaling data...")
	scaler = StandardScaler()
	X_meta = meta.as_matrix()
	X_meta = scaler.fit_transform(X_meta)

	X_meta = pd.DataFrame(X_meta)
	X_meta['target'] = train_stack.author
	X_meta.to_csv("meta_train.csv", sep=",", index=False)

if __name__ in "__main__":
	main()