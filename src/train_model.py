import sys
import os
import pickle
from time import time

from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import ShuffleSplit

from dataset import get_sentences
from feature_extraction import SpacyFeaturizer
from ploting import plot_learning_curve

"""
def number_normalizer(tokens):
	return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
"""
featurizer = SpacyFeaturizer()

class CustomLinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return featurizer.featurize(data)

def train_model():
	# Move somewhere else
	POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
	data = get_sentences(POS_SENTENCES_FOLDER)

	X = [e['sentence'] for e in data]
	y = [e['target'] for e in data]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	names = [
		"SGD","Logistic Regression", "MultinomialNB", 
		"Random Forest","Bernoulli NB", "RidgeClassifier","KNN"
	]		

	classifiers = [
		SGDClassifier(class_weight="balanced"),
		LogisticRegression(),
		MultinomialNB(),
		RandomForestClassifier(),
		BernoulliNB(),
		RidgeClassifier(),
		KNeighborsClassifier()
	]

	for name, classifier in zip(names, classifiers):
		pipe = Pipeline([
				('features', CustomLinguisticFeatureTransformer()),
				('dict_vect', DictVectorizer()),
				('classifier', classifier)
		])

		# model generation
		pipe.fit(X_train, y_train)

		# Predicting with a test dataset
		predicted = pipe.predict(X_test)
		
		print("*****************************")
		print(name)
		# Model Accuracy
		print("Accuracy:",metrics.accuracy_score(y_test, predicted))
		print("Precision:",metrics.precision_score(y_test, predicted, pos_label='fact-checkable'))
		print("Recall:",metrics.recall_score(y_test, predicted, pos_label='fact-checkable'))

		cm = confusion_matrix(y_test, predicted)
		print("CONFUSION MATRIX:")
		print(cm)

		cr = classification_report(y_test, predicted)
		print("REPORT:")
		print(cr) 
			
		title = name
		# Cross validation with 100 iterations to get smoother mean test and train
		# score curves, each time with 20% data randomly selected as a validation set.
		cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

		plt = plot_learning_curve(pipe, title, X_train, y_train, ylim=(0.7, 1.01),
							cv=cv, n_jobs=4)

		img_path = os.path.join('..','data','images',name + ".png")
		plt.savefig(img_path)

		""" FINETUNING
		from sklearn.model_selection import GridSearchCV
		parameters = {
			'vect__ngram_range': [(1, 1), (1, 2)],
			'tfidf__use_idf': (True, False),
			'clf__alpha': (1e-2, 1e-3),
		}
		gs_clf_svm = GridSearchCV(pipe, parameters, n_jobs=-1)
		gs_clf_svm = gs_clf_svm.fit(X_test, y_test)
		print(gs_clf_svm.best_score_)
		print(gs_clf_svm.best_params_)
		"""
		
		# Save to file in the current working directory
		MODELS_FOLDER = os.path.join('..','data','models')
		MODELS_PATH = os.path.join(MODELS_FOLDER, name+".pkl")

		with open(MODELS_PATH, 'wb') as file:
			pickle.dump(pipe, file)

if __name__=="__main__":
	train_model()