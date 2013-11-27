#!/usr/bin/python2

# This is a Gradient Boost Classifier

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.ensemble import GradientBoostingClassifier

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
    classifier = GradientBoostingClassifier()
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Gradient Boost Classifier"
    suite(Yv, Classifier.predict(Xv))
