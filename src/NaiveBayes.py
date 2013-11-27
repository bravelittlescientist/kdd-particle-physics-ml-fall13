#!/usr/bin/python2

# This is a Naive Bayes Classifier

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.naive_bayes import GaussianNB

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
    classifier = GaussianNB()
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Naive Bayes Classifier"
    suite(Yv, Classifier.predict(Xv))
