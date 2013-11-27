#!/usr/bin/python2

# This is a random forest ensemble classifier based on the scikit
# ensembles module.
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.
#
# Decision tree documentation:
# http://scikit-learn.org/stable/modules/tree.html

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.ensemble import RandomForestClassifier

def train(Xtrain, Ytrain, n=350):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    forest -- A random forest of n estimators, fitted to Xtrain and Ytrain
    """
    forest = RandomForestClassifier(n_estimators=n, max_depth=None, random_state=0, min_samples_split=1, max_features=9)
    forest.fit(Xtrain, Ytrain)
    return forest

if __name__ == "__main__":
    # Let's take our training data and train a random forest
    # on a subset.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Random Forest Classifier"
    suite(Yv, Classifier.predict(Xv))
