from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score

import numpy as np
from scipy.stats import wilcoxon

scoring = {
    #'Accuracy': make_scorer(accuracy_score),
    'FC Recall': make_scorer(recall_score,pos_label="fact-checkable"),
    'NFC Recall': make_scorer(recall_score,pos_label="non-fact-checkable"),
    'MACRO Recall': make_scorer(recall_score,average="macro"),
    'FC Precision': make_scorer(precision_score,pos_label="fact-checkable"),
    'NFC Precision': make_scorer(precision_score,pos_label="non-fact-checkable"),
    'MACRO Precision': make_scorer(precision_score,average="macro"),
    'FC F-measure': make_scorer(f1_score,pos_label="fact-checkable"),
    'NFC F-measure': make_scorer(f1_score,pos_label="non-fact-checkable"),
    'MACRO F-measure': make_scorer(f1_score,average="macro"),
}

def show_most_informative_features(vectorizer,selector, clf, n=50):
    """ UGLY FUNCTION TO SHOW MOST INFORMATIVE FEATURES """
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

def evaluate_classifiers(feats_vector, y, classifiers):
    print("LENN " + str(len(y)))
    print("Evaluation classifiers with a 5 fold cross validation...\n")
    for classifier in classifiers:
        #y = [int(f=="fact-checkable") for f in y]
        scores = cross_validate(classifier, feats_vector, y, scoring=scoring)

        print()
        print("\t")
        print("\t-----------------------------")
        print("\tMétrica              Promedio Desvío")
        for key, value in scores.items():
            print("%0.2f\t%0.2f" % (np.mean(value), np.std(value)))
        print("\t-----------------------------")
        print()


def evaluate_pipeline(X, y, pipe):
    print("Evaluation classifiers with a 10 fold cross validation...\n")
    cv = KFold(n_splits=(3))
    #y = [int(f=="fact-checkable") for f in y]
    scores = cross_validate(pipe, X, y, scoring=scoring)

    print()
    print("\t-----------------------------")
    print("\tMétrica      Promedio Desvío")
    for key, value in scores.items():
        print("\t%-15s\t%0.2f\t%0.2f" % (key, np.mean(value), np.std(value)))
    print("\t-----------------------------")
    print()

def paired_test(clf1, clf2, X, y):
    # Calculate p-value
    data1 = cross_val_score(clf1,X,y,scoring=make_scorer(recall_score,pos_label="fact-checkable"),cv=10)
    data2 = cross_val_score(clf2,X,y,scoring=make_scorer(recall_score,pos_label="fact-checkable"),cv=10)

    stat, p = wilcoxon(data1, data2)

    print("RECALL STAT: " + str(stat))
    print("RECALL P: " + str(p))

    data1 = cross_val_score(clf1,X,y,scoring=make_scorer(precision_score,pos_label="fact-checkable"),cv=10)
    data2 = cross_val_score(clf2,X,y,scoring=make_scorer(precision_score,pos_label="fact-checkable"),cv=10)

    stat, p = wilcoxon(data1, data2)

    print("PRECISION STAT: " + str(stat))
    print("PRECISION P: " + str(p))
    return stat, p

