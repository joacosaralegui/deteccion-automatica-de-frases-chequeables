import sys
import os
import pickle
from time import time
import numpy as np

from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, make_scorer

from dataset import get_sentences_csv as get_sentences
from feature_extraction import SpacyFeaturizer
from ploting import plot_learning_curve

import eli5

scoring = {'accuracy': make_scorer(accuracy_score),
           'recall': make_scorer(recall_score),
           'precision': make_scorer(precision_score),
           'roc_auc': make_scorer(roc_auc_score)}
"""
def number_normalizer(tokens):
	return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
"""
def show_most_informative_features(vectorizer,selector, clf, n=50):
	feature_names = vectorizer.get_feature_names()

	indexes = selector.get_support(indices=True)
	feats = []
	for i, name in enumerate(feature_names):
		if i in indexes:
			feats.append(name)
	coefs_with_fns = sorted(zip(clf.coef_[0], feats))
	top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		print("\t%.4f\t%-30s\t\t%.4f\t%-30s" % (coef_1, fn_1, coef_2, fn_2))


class SpacyFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)

def predict(sentences, featurizer, vectorizer, selector, classifier):
	# Predicting with a test dataset
	test_features = featurizer.transform(sentences)
	feat_test_dict = vectorizer.transform(test_features)
	feat_selected = selector.transform(feat_test_dict)
	return classifier.predict(feat_selected)

def train_model(featurizer, vectorizer, names, classifiers):
	# Move somewhere else
	POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
	data = get_sentences(POS_SENTENCES_FOLDER)

	X = [e['sentence'] for e in data]
	y = [e['target'] for e in data]
	#Select top 2 features based on mutual info regression

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

	selector = SelectKBest(chi2, k = 500)
	
	print("Extracting features...")
	features = featurizer.fit_transform(X_train)
	print("Feature extraction endend!")

	# Fit transform vectorizer
	features_dict = vectorizer.fit_transform(features)
	features_dict = selector.fit_transform(features_dict, y_train)

	for name, classifier in zip(names, classifiers):
		y_train_label_encoded = [int(label=="fact-checkable") for label in y_train]
		scores = cross_validate(classifier, features_dict, y_train_label_encoded, cv=5,scoring=scoring)
		import pdb;pdb.set_trace()
		for key, value in scores.items():
			print("%0.2f %s with a standard deviation of %0.2f" % (np.mean(value), key, np.std(value)))
		# model generation
		classifier.fit(features_dict, y_train)

		print("Classifier fitted")


		predicted = predict(X_test, featurizer, vectorizer, selector, classifier)	
		# See the mistakes made
		"""
		print("********** PREDICTION MISTAKES ********")
		for i,prediction in enumerate(predicted):
			if(prediction != y_test[i]):
				print(X_test[i] + ": "+ prediction)
		print("*****************************")
		"""
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
			
		show_features = True

		if show_features:
			try:
				#show_most_informative_features(pipe['dict_vect'],pipe['classifier'])
				show_most_informative_features(vectorizer,selector,classifier)
			except Exception as e:
				print(e)
				print("Skip features")
		"""	
		title = name
		# Cross validation with 100 iterations to get smoother mean test and train
		# score curves, each time with 20% data randomly selected as a validation set.
		cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

		plt = plot_learning_curve(pipe, title, X_train, y_train, ylim=(0.7, 1.01),
							cv=cv, n_jobs=4)

		img_path = os.path.join('..','data','images',name + ".png")
		plt.savefig(img_path)
		"""
		"""
		from sklearn.model_selection import GridSearchCV
		parameters = {
			'classifier__alpha': (1e-2, 1e-3),
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
			pickle.dump(classifier, file)

if __name__=="__main__":

	featurizer = SpacyFeatureTransformer()
	vectorizer = DictVectorizer()

	names = [
		"SGD",
		"Logistic Regression", 
		"MultinomialNB", 
		#"Random Forest",
		#"Bernoulli NB", 
		#"RidgeClassifier",
		#"KNN",
		#"SVM"
	]		

	classifiers = [
		SGDClassifier(class_weight="balanced"),
		LogisticRegression(class_weight="balanced",max_iter=400),
		MultinomialNB(alpha=0.01),
		#RandomForestClassifier(),
		#BernoulliNB(),
		#RidgeClassifier(),
		#KNeighborsClassifier(),
		#SVC()
	]

	print("*************** CUSTOM FEATURES  ************************")
	train_model(featurizer, vectorizer, names, classifiers)

	#featurizer = CountVectorizer()
	#vectorizer = TfidfTransformer()

	#print("*************** TFIDF  ************************")
	#train_model(featurizer, vectorizer, names, classifiers)