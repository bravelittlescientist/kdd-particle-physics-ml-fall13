#!/usr/bin/python2

# This is an extra random trees ensemble classifier based on the scikit
# ensembles module.

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV

def train(Xtrain, Ytrain, n=250, d=None):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier
    """
    classifier = ExtraTreesClassifier(n_estimators=n, max_depth=d, min_samples_split = 1, random_state=0, max_features=36)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a random forest
    # on a subset.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Extra Random Trees Ensemble Classifier"
    suite(Yv, Classifier.predict(Xv))
