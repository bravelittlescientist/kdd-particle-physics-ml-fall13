#!/usr/bin/python2

# This is a performance test to capture our baseline classification data.

import sys

from sklearn.cross_validation import cross_val_score
from util import load_validation_data

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    # Get training data
    Xt, Yt, Xunused = load_validation_data()

    # Cross validation, 5-fold
    cvf = 5

    # Initialize classifiers
    classifiers = {
        "Naive Bayes"         : GaussianNB(),
        "Gradient Boost"      : GradientBoostingClassifier(),
        "Adaboost"            : AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
        "Decision Tree"       : DecisionTreeClassifier(),
        "Extra Random Trees"  : ExtraTreesClassifier(n_estimators=300),
        "Logistic Regression" : LogisticRegression(),
        "K-Nearest-Neighbors" : KNeighborsClassifier(),
        "SGD"                 : SGDClassifier(),
        "SVM"                 : LinearSVC(),
        "Random Forest"       : RandomForestClassifier(n_estimators=300)
    }

    for c in classifiers:
      # cross validation
      scores = cross_val_score(classifiers[c], Xt, Yt, cv=cvf)

      # report
      print c,scores.mean()
