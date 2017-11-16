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

def main():
	## load test data
	test = pd.read_csv("meta_test.csv", sep=",")

	## isolate id and target
	idVal = test.id

	## convert inputs to 2x2 array
	X = test.drop(["id"], axis=1).as_matrix()

	## load models
	knn = pickle.load(open("knn.pkl", "rb"))
	gbc = pickle.load(open("gbc.pkl", "rb"))

	## predict proba and store
	authCol = ["EAP", "HPL", "MWS"]
	knn_prob = pd.concat([pd.DataFrame(idVal), pd.DataFrame(knn.predict_proba(X), columns=authCol)], axis=1)
	gbc_prob = pd.concat([pd.DataFrame(idVal), pd.DataFrame(gbc.predict_proba(X), columns=authCol)], axis=1)

	knn_prob.to_csv("predictions/knn_pred.csv", sep=",", index=False)
	gbc_prob.to_csv("predictions/gbc_pred.csv", sep=",", index=False)

if __name__ in "__main__":
	main()
