#!/usr/bin/python2

# This is an extra random trees ensemble classifier based on the scikit
# ensembles module.

import sys

from imputation import load_data
from util import shuffle_split
from metrics import suite

from sklearn.ensemble import ExtraTreesClassifier

def train(Xtrain, Ytrain, n=100, d=None):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier
    """
    classifier = ExtraTreesClassifier(n_estimators=n, max_depth=d, min_samples_split = 1, random_state=0)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a random forest
    # on a subset.

    if len(sys.argv) < 2:
        print "Usage: $ python decision-tree.py /path/to/data/file/"
    else:
        # Obtain and split/shuffle our training data
        training = sys.argv[1]
        X,Y,n,f = load_data(training)
        Xt, Xv, Yt, Yv = shuffle_split(X,Y)

        # Train a forest on it
        Classifier = train(Xt, Yt)
        print "Extra Random Trees Ensemble Classifier"
        suite(Yv, Classifier.predict(Xv))
