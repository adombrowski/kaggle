import pandas as pd
import numpy as np
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag

def clean(sentences, as_list=False):
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
		tags = pos_tag(i)
		for t in tags:
			if row.get(t[1], None) is not None:
				row[t[1]] += 1
			else:
				row[t[1]] = 1
		row = pd.DataFrame(pd.Series(row)).T
		results = results.append(row)
	return results.fillna(0).reset_index(drop=True)

def printConfusion(y_act, y_pred):
	cm = confusion_matrix(y_act, y_pred)
	cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
	print(pd.DataFrame(cm))
	print(pd.DataFrame(cm_norm))

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
	print "Cleaning..."
	train['clean_text'] = clean(train.text)
	print "Clean complete..."

	"""
	Step 1:
		Train base tfidf classifiers, mnb
	"""

	## initialize tfidf vectorizer and transform data
	print "Training MNB..."
	tfidf_vectorizer = TfidfVectorizer(max_features=4000).fit(train.clean_text)
	tfidf = tfidf_vectorizer.transform(train.clean_text)

	## set targets
	y = train.author.as_matrix().ravel()

	## set split
	X_train, X_test, y_train, y_test = train_test_split(
		tfidf, y, random_state=0, stratify=y)

	## train base classifiers
	mnb = MultinomialNB(alpha=.1).fit(X_train, y_train)
	print("Training set score: {:.3f}".format(mnb.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(mnb.score(X_test, y_test)))
	print("MNB F1 Score: {:.3f}".format(f1_score(y_test, mnb.predict(X_test), average='weighted')))
	printConfusion(y_test, mnb.predict(X_test))

	"""
	Step 2:
		Prepare and train stack classifier
	"""

	## generate pos data
	print "Calculating POS..."
	pos = posCount(train.text)
	print "Running SVD on POS data..."
	pos = TruncatedSVD(n_components=2).fit_transform(pos.as_matrix())

	## store in df
	meta = pd.concat([pd.DataFrame(pos), pd.DataFrame(mnb.predict_proba(tfidf))], axis=1)

	## calculate char count
	meta['char_count'] = characterCount(train.clean_text)

	## tag unique
	meta = pd.concat([meta, tagUnique(train)], axis=1)

	## scale data
	print "Scaling data..."
	scaler = StandardScaler()
	X_meta = meta.as_matrix()
	X_meta = scaler.fit_transform(X_meta)

	X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
		X_meta, y, random_state=0, stratify=y)


	print "Training KNN..."
	knn = KNeighborsClassifier(n_neighbors=20).fit(X_train_meta, y_train_meta)
	print("Training set score: {:.3f}".format(knn.score(X_train_meta, y_train_meta)))
	print("Test set score: {:.3f}".format(knn.score(X_test_meta, y_test_meta)))
	print("KNN F1 Score: {:.3f}".format(f1_score(y_test_meta, knn.predict(X_test_meta), average='weighted')))

	print "Training GBC..."
	gbc = GradientBoostingClassifier().fit(X_train_meta, y_train_meta)
	print("Training set score: {:.3f}".format(gbc.score(X_train_meta, y_train_meta)))
	print("Test set score: {:.3f}".format(gbc.score(X_test_meta, y_test_meta)))
	print("GBC F1 Score: {:.3f}".format(f1_score(y_test_meta, gbc.predict(X_test_meta), average='weighted')))


if __name__ in "__main__":
	main()