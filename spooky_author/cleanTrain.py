import pandas as pd
import numpy as np
import pickle
import json
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from utils import *

def calcUnique(text, target):
	## set unique
	eap = [item for sublist in [word_tokenize(c) for c in text[target=="EAP"]] for item in sublist]
	hpl = [item for sublist in [word_tokenize(c) for c in text[target=="HPL"]] for item in sublist]
	mws = [item for sublist in [word_tokenize(c) for c in text[target=="MWS"]] for item in sublist]

	return [
		list(set(findUnique(eap, set(hpl).union(set(mws))))),
		list(set(findUnique(hpl, set(eap).union(set(mws))))),
		list(set(findUnique(mws, set(hpl).union(set(eap)))))
	]

def main():
	## import train.csv
	train = pd.read_csv("train.csv", sep=",")

	## clean text samples
	print("Cleaning...")
	train['clean_text'] = clean(train.text)
	train['starts'] = padSingle(startWords(train.text))
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
	pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))

	tfidf_base = tfidf_vectorizer.transform(train_base.clean_text)
	tfidf_stack = tfidf_vectorizer.transform(train_stack.clean_text)

	## set targets
	y = train_base.author.as_matrix().ravel()

	## train base classifiers
	mnb = MultinomialNB(alpha=.1).fit(tfidf_base, y)
	pickle.dump(mnb, open("mnb.pkl", "wb"))

	"""
	Step 2:
		Prepare data for stack classifier
	"""

	## generate pos data
	"""
	print("Calculating POS...")
	pos = posCount(train_stack.text)
	print("Running SVD on POS data...")
	pos = TruncatedSVD(n_components=2).fit_transform(pos.as_matrix())

	## store in df
	meta = pd.concat([pd.DataFrame(pos), pd.DataFrame(mnb.predict_proba(tfidf_stack))], axis=1)
	"""
	meta = pd.DataFrame(mnb.predict_proba(tfidf_stack))
	## calculate char count
	meta['char_count'] = characterCount(train_stack.clean_text)
	## add special character data
	meta['char_count'] = characterCount(train_stack.clean_text)
	meta['n_;'] = itemCount(train_stack.text, ";")
	meta['n_,'] = itemCount(train_stack.text, ",")
	meta['n_...'] = itemCount(train_stack.text, "...")
	## add higher level language features
	meta['allit'] = alliteration(train_stack.clean_text)
	meta['asson'] = assonance(train_stack.clean_text)
	meta['vowRatio'] = vowelRatio(train_stack.clean_text)
	meta['conj'] = conjunction(train_stack.text)
	meta['rep'] = repetition(train_stack.clean_text)
	#meta['stop'] = stopCount(train_stack.text)

	## tag unique
	uniq_tok = calcUnique(train_base.clean_text, train_base.author)
	uniq_start = calcUnique(train_base.clean_text, train_base.author)
	json.dump(uniq_tok, open("uniq_tok.json", "w"))
	json.dump(uniq_start, open("uniq_start.json", "w"))
	meta = pd.concat([meta, tagUnique(train_stack.clean_text, uniq_tok)], axis=1)
	meta = pd.concat([meta, tagUnique(train_stack.clean_text, uniq_start)], axis=1)

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