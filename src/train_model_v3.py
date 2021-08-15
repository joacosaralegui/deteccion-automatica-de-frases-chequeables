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
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler

from ploting import plot_learning_curve

# Our imports
import evaluation
from ploting import plot_learning_curve
from dataset import get_sentences_csv as get_sentences
from feature_extraction import SpacyFeatureTransformer, EmbeddingsFeatureTransformer, SpacyFeatureTransformerFF, SpacyFeatureTransformerChq


def evaluate_classifiers(featurizer, vectorizer, selector, classifiers):
    POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
    data = get_sentences(POS_SENTENCES_FOLDER)

    X = [e['sentence'] for e in data]
    y = [int(e['target']=="fact-checkable") for e in data]

    #feats_vector = process_features(X, featurizer, vectorizer, selector)

    #evaluation.evaluate_classifiers(feats_vector, y, classifiers)


def train_models(transformers, classifiers):

    POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
    data = get_sentences(POS_SENTENCES_FOLDER)

    X = [e['sentence'] for e in data]
    y = [e['target'] for e in data]    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    evaluation.evaluate_classifiers(X_train,y_train,[Pipeline([('transformers',transformers),('clf',c)]) for _,c in classifiers])
    

    for name, classifier in classifiers:
        pipe = Pipeline([
                ('transformers', transformers),
                ('classifier', classifier),
        ])

        print(name)
        # model generation
        pipe.fit(X_train, y_train)    
        
        #evaluation.evaluate_pipeline(X, y, pipe)
        #evaluation.evaluate_classifiers(transformers.fit_transform(X,y),[classifier])
        
        # Predicting with a test dataset
        predicted = pipe.predict(X_test)
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
        """
        title = name
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

        plt = plot_learning_curve(pipe, title, X, y, ylim=(0.7, 1.01),
                            cv=cv, n_jobs=1)

        img_path = os.path.join('..','data','images',name + ".png")
        plt.savefig(img_path)
        # """
        cr = classification_report(y_test, predicted)
        print("REPORT:")
        print("---------------------------------------------")
        print(cr) 
        print("---------------------------------------------")
        
        show_features = True

        if show_features:
            try:
                #show_most_informative_features(pipe['dict_vect'],pipe['classifier'])
                evaluation.show_most_informative_features(transformers,classifier)
            except Exception as e:
                print(e)
                print("Skip features")

        # Save to file in the current working directory
        MODELS_FOLDER = os.path.join('..','data','models')
        MODELS_PATH = os.path.join(MODELS_FOLDER, name+".pkl")

        with open(MODELS_PATH, 'wb') as file:
            pickle.dump(classifier, file)

if __name__=="__main__":
    transformers =  FeatureUnion(
        transformer_list=[
            # Pipeline for pulling features from the post's subject line
            ('spacy', Pipeline([
               ('features', SpacyFeatureTransformer()),
               ('vectorizer', DictVectorizer()),
               ('selector', SelectPercentile(chi2, percentile=50))
            ])),

            # Pipeline for standard bag-of-words model for body
            #('tfidf', Pipeline([
            #   ('features', CountVectorizer()),
            #   ('tfidf',  TfidfTransformer()),
            #])),

            # Pipeline for standard bag-of-words model for body
            ('embeddings', Pipeline([
               ('features', EmbeddingsFeatureTransformer()),
               ('scaler', MinMaxScaler()),
            ])),
    ])

    transformers_chq = Pipeline([
               ('features', SpacyFeatureTransformerChq()),
               ('vectorizer', DictVectorizer()),
               ('selector', SelectPercentile(chi2, percentile=50))
            ])
    
    transformers_ff = FeatureUnion(
        transformer_list=[

            # Pipeline for standard bag-of-words model for body
            ('embeddings', Pipeline([
                ('features', EmbeddingsFeatureTransformer()),
                ('scaler', MinMaxScaler())
            ])),
    ])

    classifiers = [
        ("SGD", SGDClassifier(class_weight="balanced")),
        ("Logistic Regression", LogisticRegression(class_weight="balanced",max_iter=1000)),
        ("MultinomialNB", MultinomialNB(alpha=0.01)),
        #("DecisionTree", DecisionTreeClassifier()),
        #("ComplementNB", ComplementNB()),
        #("GaussianNB", GaussianNB()),
        ("Random Forest",RandomForestClassifier()),
        #("Bernoulli NB", BernoulliNB()),
        #("RidgeClassifier",RidgeClassifier()),
        ("SVM", SVC())
    ]	

    print("*************** TRAIN EACH MODEL ************************")
    print()
    #train_models(transformers_chq, classifiers)
    #train_models(transformers, classifiers)
    train_models(transformers_ff, classifiers)
    print()

    #print("*************** TFIDF  ************************")
    #train_model(featurizer, vectorizer, names, classifiers)