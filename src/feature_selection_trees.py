#!/usr/bin/python2

# This is an extra random trees ensemble classifier based on the scikit
# ensembles module.

import sys

import numpy as np

from util import get_split_training_dataset
from metrics import suite

from sklearn.ensemble import ExtraTreesClassifier

def get_important_features(Xtrain, Ytrain, n=250, d=None, threshold=0.01):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    ranking -- a ranked list of indices of important features
    """
    # Train and fit tree classifier ensemble
    classifier = ExtraTreesClassifier(n_estimators=n, random_state=0)
    classifier.fit(Xtrain, Ytrain)

    # Compute important features
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    ranking = [[indices[f], importances[indices[f]]] for f in range(Xtrain.shape[1])]

    ranking = filter(lambda r: r[1] >= threshold, ranking)

    return ranking

if __name__ == "__main__":
    # Let's take our training data and use a forest
    # to select the best features...
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    ranking = get_important_features(Xt, Yt, threshold=0)

    print "Feature ranking:"
    for r in range(len(ranking)):
        print str(r+1) + ". ", ranking[r][0], ranking[r][1]

