# Generic imports
import os
import pickle
import numpy as np
from time import time
from nltk.corpus import stopwords

# Sklearn imports
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.svm import SVC

# Our imports
import evaluation
from ploting import plot_learning_curve
from dataset import get_sentences_csv as get_sentences
from feature_extraction import CustomLinguisticFeatureTransformer

def process_features(sentences, featurizer, vectorizer, selector):
    """
    Pass sentences through pipelinen to get selected features vector
    """
    test_features = featurizer.transform(sentences)
    feat_test_dict = vectorizer.transform(test_features)
    feat_selected = selector.transform(feat_test_dict)
    return feat_selected

def predict(sentences, featurizer, vectorizer, selector, classifier):
    """
    Predict a list of sentences as fact-checkable or not
    """
    return classifier.predict(process_features(sentences,featurizer,vectorizer,selector))

def evaluate_classifiers(featurizer, vectorizer, selector, classifiers):
    POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
    data = get_sentences(POS_SENTENCES_FOLDER)

    X = [e['sentence'] for e in data]
    y = [int(e['target']=="fact-checkable") for e in data]

    feats_vector = process_features(X, featurizer, vectorizer, selector)

    evaluation.evaluate_classifiers(feats_vector, y, classifiers)


def train_models(featurizer, vectorizer, selector, classifiers):

    POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
    data = get_sentences(POS_SENTENCES_FOLDER)

    X = [e['sentence'] for e in data]
    y = [e['target'] for e in data]    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

    print("Extracting features...")
    features = featurizer.fit_transform(X_train)
    print("Feature extraction endend!")

    # Fit transform vectorizer
    features_dict = vectorizer.fit_transform(features)
    features_dict = selector.fit_transform(features_dict, y_train)

    for name, classifier in classifiers:
        print()
        print("Training " + name+ " ...")
        # model generation
        classifier.fit(features_dict, y_train)

        predicted = predict(X_test, featurizer, vectorizer, selector, classifier)	
        # See the mistakes made
        """
        print("********** PREDICTION MISTAKES ********")
        for i,prediction in enumerate(predicted):
            if(prediction != y_test[i]):
                print(X_test[i] + ": "+ prediction)
        print("*****************************")
        """
        """        
        # Model Accuracy
        print("Accuracy:",metrics.accuracy_score(y_test, predicted))
        print("Precision:",metrics.precision_score(y_test, predicted, pos_label='fact-checkable'))
        print("Recall:",metrics.recall_score(y_test, predicted, pos_label='fact-checkable'))

        cm = confusion_matrix(y_test, predicted)
        print("CONFUSION MATRIX:")
        print(cm)
        """

        cr = classification_report(y_test, predicted)
        print("REPORT:")
        print("---------------------------------------------")
        print(cr) 
        print("---------------------------------------------")
            
        show_features = False

        if show_features:
            try:
                #show_most_informative_features(pipe['dict_vect'],pipe['classifier'])
                evaluation.show_most_informative_features(vectorizer,selector,classifier)
            except Exception as e:
                print(e)
                print("Skip features")

        # Save to file in the current working directory
        MODELS_FOLDER = os.path.join('..','data','models')
        MODELS_PATH = os.path.join(MODELS_FOLDER, name+".pkl")

        with open(MODELS_PATH, 'wb') as file:
            pickle.dump(classifier, file)

if __name__=="__main__":
    featurizer = CustomLinguisticFeatureTransformer()
    vectorizer = DictVectorizer()
    selector = SelectKBest(chi2, k = 1000)

    classifiers = [
        ("SGD", SGDClassifier(class_weight="balanced")),
        ("Logistic Regression", LogisticRegression(class_weight="balanced",max_iter=400)),
        ("MultinomialNB", MultinomialNB(alpha=0.01)),
        ("DecisionTree", DecisionTreeClassifier()),
        #("ComplementNB", ComplementNB()),
        #("GaussianNB", GaussianNB()),
        ("Random Forest",RandomForestClassifier()),
        ("Bernoulli NB", BernoulliNB()),
        ("RidgeClassifier",RidgeClassifier()),
        ("SVM", SVC())
    ]	

    print("*************** TRAIN EACH MODEL ************************")
    print()
    train_models(featurizer, vectorizer, selector, classifiers)
    print()

    print("*************** MODELS EVALUATION ********************")
    evaluate_classifiers(featurizer, vectorizer, selector, classifiers)
    #featurizer = CountVectorizer()
    #vectorizer = TfidfTransformer()

    #print("*************** TFIDF  ************************")
    #train_model(featurizer, vectorizer, names, classifiers)