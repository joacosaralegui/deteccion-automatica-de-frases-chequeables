from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, make_scorer
import numpy as np

scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'Recall': make_scorer(recall_score),
    'Precision': make_scorer(precision_score),
    'F-measure': make_scorer(f1_score),
    'ROC-AUC': make_scorer(roc_auc_score)
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
    print("Evaluation classifiers with a 10 fold cross validation...\n")
    for name, classifier in classifiers:
        scores = cross_validate(classifier, feats_vector, y, cv=10, scoring=scoring)

        print()
        print("\t" + name)
        print("\t-----------------------------")
        print("\tMétrica      Promedio Desvío")
        for key, value in scores.items():
            print("\t%-15s\t%0.2f\t%0.2f" % (key, np.mean(value), np.std(value)))
        print("\t-----------------------------")
        print()