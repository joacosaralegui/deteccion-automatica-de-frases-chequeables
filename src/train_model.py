# Generic imports
from datetime import  datetime
import os
import pickle

# Sklearn imports
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler

# Project imports
import evaluation
from dataset import get_sentences_csv 
from feature_extraction import TraditionalFeatureTransformer, EmbeddingsFeatureTransformer

def train_models(transformers, classifiers):
    """
    Entrena las diferentes combinaciones de transformers y clasificadores y genera reportes de cada una
    """
    POS_SENTENCES_FOLDER = os.path.join('..','data','tagged_corpus')
    data = get_sentences_csv(POS_SENTENCES_FOLDER)

    X = [e['sentence'] for e in data]
    y = [e['target'] for e in data]    
    
    # Tomo la data de evaluación para evaluar la performance de los modelos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    evaluation.evaluate_classifiers(X_train,y_train,[Pipeline([('transformers',transformers),('clf',c)]) for _,c in classifiers])
    
    # Itero sobre las variantes
    for name, classifier in classifiers:
        pipe = Pipeline([
                ('transformers', transformers),
                ('classifier', classifier),
        ])

        print(name)

        # model generation
        pipe.fit(X_train, y_train)    

        # Tomo el tiempo de prediccion        
        print("Cantidad de frases: " + str(len(X_test)))
        time = datetime.now()
        # Predicting with a test dataset
        predicted = pipe.predict(X_test)
        print("Tardo: " + str(datetime.now()-time))
        
        
        # Clasification report        
        cr = classification_report(y_test, predicted)
        print("REPORT:")
        print("---------------------------------------------")
        print(cr) 
        print("---------------------------------------------")

        # Save to file in the current working directory
        MODELS_FOLDER = os.path.join('..','data','models')
        MODELS_PATH = os.path.join(MODELS_FOLDER, name+".pkl")

        with open(MODELS_PATH, 'wb') as file:
            pickle.dump(classifier, file)

if __name__=="__main__":
    transformers_completo =  FeatureUnion(
        transformer_list=[
            # Pipeline for pulling features from the post's subject line
            ('spacy', Pipeline([
               ('features', TraditionalFeatureTransformer()),
               ('vectorizer', DictVectorizer()),
               ('selector', SelectPercentile(chi2, percentile=50))
            ])),

            # Pipeline for standard bag-of-words model for body
            ('embeddings', Pipeline([
               ('features', EmbeddingsFeatureTransformer()),
               ('scaler', MinMaxScaler()),
            ])),
    ])

    transformers_tradicionales = Pipeline([
               ('features', TraditionalFeatureTransformer()),
               ('vectorizer', DictVectorizer()),
               ('selector', SelectPercentile(chi2, percentile=50))
            ])
    
    transformers_vectoriales = FeatureUnion(
        transformer_list=[
            # Pipeline for standard bag-of-words model for body
            ('embeddings', Pipeline([
                ('features', EmbeddingsFeatureTransformer()),
                ('scaler', MinMaxScaler())
            ])),
    ])

    classifiers = [
        ("Logistic Regression", LogisticRegression(class_weight="balanced",max_iter=1000)),
        ("MultinomialNB", MultinomialNB(alpha=0.01)),
        ("Random Forest",RandomForestClassifier()),
        ("SVM", SVC())
    ]	

    print("*************** TRAIN EACH MODEL ************************")
    print()
    train_models(transformers_tradicionales, classifiers)
    train_models(transformers_vectoriales, classifiers)
    train_models(transformers_completo, classifiers)
    print()
