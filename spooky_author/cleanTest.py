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

def main():
	## import test data
	test = pd.read_csv("test.csv", sep=",")

	## clean text samples
	print("Cleaning...")
	test['clean_text'] = clean(test.text)
	test['starts'] = padSingle(startWords(test.text))
	print("Clean complete...")

	## tfidf vectorize
	tfidf_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
	tfidf = tfidf_vectorizer.transform(test.clean_text)

	## return mnb predicted probabilities
	mnb = pickle.load(open("mnb.pkl", "rb"))
	mnb_prob = mnb.predict_proba(tfidf)

	## calculate meta-data fields
	## generate pos data
	"""
	print("Calculating POS...")
	pos = posCount(test.text)
	print("Running SVD on POS data...")
	pos = TruncatedSVD(n_components=2).fit_transform(pos.as_matrix())
	"""

	## store in df
	#meta = pd.concat([pd.DataFrame(pos), pd.DataFrame(mnb_prob)], axis=1)
	meta = pd.DataFrame(mnb_prob)

	## calculate char count
	meta['char_count'] = characterCount(test.clean_text)

	meta['char_count'] = characterCount(test.clean_text)
	meta['n_;'] = itemCount(test.text, ";")
	meta['n_,'] = itemCount(test.text, ",")
	meta['n_...'] = itemCount(test.text, "...")
	## add higher level language features
	meta['allit'] = alliteration(test.clean_text)
	meta['asson'] = assonance(test.clean_text)
	meta['vowRatio'] = vowelRatio(test.clean_text)
	meta['conj'] = conjunction(test.text)
	meta['rep'] = repetition(test.clean_text)
	#meta['stop'] = stopCount(test.text)

	## tag unique
	uniq_tok = json.load(open("uniq_tok.json", "r"))
	uniq_stop = json.load(open("uniq_start.json", "r"))
	meta = pd.concat([meta, tagUnique(test.clean_text, uniq_tok)], axis=1)
	meta = pd.concat([meta, tagUnique(test.clean_text, uniq_stop)], axis=1)

	## scale data
	print("Scaling data...")
	scaler = StandardScaler()
	X_meta = meta.as_matrix()
	X_meta = scaler.fit_transform(X_meta)

	X_meta = pd.DataFrame(X_meta)
	X_meta['id'] = test.id
	X_meta.to_csv("meta_test.csv", sep=",", index=False)

if __name__ in "__main__":
	main()