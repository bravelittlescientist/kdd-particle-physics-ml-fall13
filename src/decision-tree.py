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

from sklearn import tree
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def acc_training_split_shuffle_predict(X,Y):
    """ Basic split and shuffle and classification of X and Y """
    # Split & shuffle
    shuffle(X,Y)
    Xt, Xv, Yt, Yv = train_test_split(X, Y, train_size=0.75)

    # Train
    clf = tree.DecisionTreeClassifier()
    clf.fit(Xt, Yt)
    error = accuracy_score(Yv, clf.predict(Xv))
    print error

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.

    if len(sys.argv) < 2:
        print "Usage: $ python decision-tree.py /path/to/data/file/"
    else:
        training = sys.argv[1]
        X,Y,n,f = load_data(training)
        acc_training_split_shuffle_predict(X,Y)


