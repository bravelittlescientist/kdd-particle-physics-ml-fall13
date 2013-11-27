#!/usr/bin/python2

# This is a NN classifier based on the scikit-learn documentation.
#
# http://scikit-learn.org/stable/modules/neighbors.html

import sys

from imputation import load_data
from util import shuffle_split
from metrics import suite

from sklearn.neighbors import KNeighborsClassifier

def train(Xtrain, Ytrain):
    """ Use entirety of provided X, Y to predict

    Default Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Named Arguments
    --

    Returns
    classifier -- a tree fitted to Xtrain and Ytrain
    """
    classifier = KNeighborsClassifier(125)
    classifier.fit(Xtrain, Ytrain)
    return classifier

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
        print "KNN Accuracy"
        suite(Yv, Classifier.predict(Xv))
