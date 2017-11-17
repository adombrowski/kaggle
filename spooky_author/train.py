import pandas as pd
import numpy as np
import pickle
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag

def printConfusion(y_act, y_pred):
	cm = confusion_matrix(y_act, y_pred)
	cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
	print(pd.DataFrame(cm))
	print(pd.DataFrame(cm_norm))

def main():
	## import train.csv
	df = pd.read_csv("meta_train.csv", sep=",")

	X, y = df.drop("target", axis=1).as_matrix(), df.target.as_matrix().ravel()

	## split training
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, random_state=0, stratify=y)

	"""
	Set parameter dictionaries
	"""

	knn = KNeighborsClassifier(n_neighbors=30).fit(X_train, y_train)
	print("Training set score: {:.3f}".format(knn.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(knn.score(X_test, y_test)))
	print("KNN F1 Score: {:.3f}".format(f1_score(y_test, knn.predict(X_test), average='weighted')))
	printConfusion(y_test, knn.predict(X_test))
	pickle.dump(knn, open("knn.pkl", "wb"))


	print("Training GBC...")
	gbc = GradientBoostingClassifier(learning_rate=.7, max_depth=1, n_estimators=100).fit(X_train, y_train)
	print("Training set score: {:.3f}".format(gbc.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(gbc.score(X_test, y_test)))
	print("GBC F1 Score: {:.3f}".format(f1_score(y_test, gbc.predict(X_test), average='weighted')))
	printConfusion(y_test, gbc.predict(X_test))
	pickle.dump(gbc, open("gbc.pkl", "wb"))

if __name__ in "__main__":
	main()