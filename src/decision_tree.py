#!/usr/bin/python2

# This is a decision tree classifier based on the scikit-learn example,
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

from sklearn import tree

def classify(Xtrain, Ytrain):
    """ Use entirety of provided X, Y to predict

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    ready_tree -- a tree fitted to Xtrain and Ytrain
    """
    ready_tree = tree.DecisionTreeClassifier()
    ready_tree.fit(Xtrain, Ytrain)
    return ready_tree

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.

    if len(sys.argv) < 2:
        print "Usage: $ python decision-tree.py /path/to/data/file/"
    else:
        training = sys.argv[1]
        X,Y,n,f = load_data(training)
        Xt, Xv, Yt, Yv = shuffle_split(X,Y)
        tree = classify(Xt, Yt)
        print "Decision Tree Accuracy:",acc(Yv, tree.predict(Xv)),"%"
