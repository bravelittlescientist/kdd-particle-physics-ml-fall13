#!/usr/bin/python2

# This is an SVM classifier, designed to
# optimize for AUC (hopefully)

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.svm import LinearSVC, SVC

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
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Get training data
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)

    # Report results
    print "SVM Classifier"
    suite(Yv, Classifier.predict(Xv))
