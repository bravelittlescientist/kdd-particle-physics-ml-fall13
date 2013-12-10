#!/usr/bin/python2

# This is a Stochastic Gradient Descent classifier based on the
# scikit-learn documentation.
#
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

import sys

from util import get_split_training_dataset
from metrics import suite

import feature_selection_trees as fclassify

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

    # smaller feature set
    Xtimp, features = fclassify.get_important_data_features(Xt, Yt, max_features=25)
    Xvimp = fclassify.compress_data_to_important_features(Xv, features)
    ClassifierImp = train(Xtimp,Yt)
    print "SGD Classiifer, 25 important features"
    suite(Yv, ClassifierImp.predict(Xvimp))
