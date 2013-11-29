#!/usr/bin/python2

# This is an Adaboost classifier

import sys

from util import get_split_training_dataset
from metrics import suite

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
    # Initialize classifier parameters for adaboost
    # For adaboost, this means the number of estimators for now
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    parameters = {'n_estimators': [150]}

    # Classify over grid of parameters
    classifier = GridSearchCV(ada, parameters)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Adaboost Classifier"
    suite(Yv, Classifier.predict(Xv))
