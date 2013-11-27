#!/usr/bin/python2

# This is an Adaboost classifier

import sys

from imputation import load_data
from util import shuffle_split
from metrics import suite

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def train(Xtrain, Ytrain):
    """ Use entirety of provided X, Y to predict

    Default Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Named Arguments
    C -- regularization parameter

    Returns
    classifier -- a tree fitted to Xtrain and Ytrain
    """
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=250)
    ada.fit(Xtrain, Ytrain)
    return ada

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.

    if len(sys.argv) < 2:
        print "Usage: $ python decision-tree.py /path/to/data/file/"
    else:
        training = sys.argv[1]
        X,Y,n,f = load_data(training)
        Xt, Xv, Yt, Yv = shuffle_split(X,Y)
        Classifier = train(Xt, Yt)
        print "Adaboost Classifier"
        suite(Yv, Classifier.predict(Xv))
