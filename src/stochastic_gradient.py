#!/usr/bin/python2

# This is a Stochastic Gradient Descent classifier based on the
# scikit-learn documentation.
#
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

import sys

from util import get_split_training_dataset
from metrics import suite

import numpy as np

from sklearn.linear_model import SGDClassifier

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
    classifier = SGDClassifier(shuffle=True, n_iter=10000)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":

    # Obtain split training data
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)

    # Report results
    print "SGD Classifier"
    suite(Yv, Classifier.predict(Xv))