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

from imputation import load_data
from util import shuffle_split
from metrics import acc

from sklearn.ensemble import RandomForestClassifier

def trainForest(Xtrain, Ytrain, n=100):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    forest -- A random forest of n estimators, fitted to Xtrain and Ytrain
    """
    forest = RandomForestClassifier(n_estimators=n)
    forest.fit(Xtrain, Ytrain)
    return forest

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
        forest = trainForest(Xt, Yt)
        accuracy = acc(Yv, forest.predict(Xv))
        print "Random Forest Ensemble Classifier:",accuracy*100,"%"
