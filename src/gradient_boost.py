#!/usr/bin/python2

# This is a Gradient Boost Classifier

import sys

from util import get_split_training_dataset, write_test_prediction, load_validation_data
from metrics import suite

from sklearn.grid_search import GridSearchCV

import feature_selection_trees as fclassify

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

def train(Xtrain, Ytrain, metric='accuracy'):
    """ Use entirety of provided X, Y to predict

    Default Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Named Arguments
    metric -- string, accuracy or roc_auc

    Returns
    classifier -- a tree fitted to Xtrain and Ytrain
    """
    gbc = GradientBoostingClassifier(verbose=1)
    parameters = {'max_depth' : range(3,11),'n_estimators' : [400,500]}

    classifier = GridSearchCV(gbc, parameters, scoring=metric)
    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a decision tree
    # on a subset. Scikit-learn provides a good module for cross-
    # validation.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "Gradient Boost Classifier"
    suite(Yv, Classifier.predict(Xv))

    # smaller feature set
    Xtimp, features = fclassify.get_important_data_features(Xt, Yt)
    Xvimp = fclassify.compress_data_to_important_features(Xv, features)
    ClassifierImp = train(Xtimp,Yt)
    print "Gradient Boosts Classiifer, 25 important features"
    suite(Yv, ClassifierImp.predict(Xvimp))

    # save predictions on test data

    X, Y, validation_data = load_validation_data()
    predictions = Classifier.predict(validation_data)
    filename = 'gradient_boost_predictions.txt'
    write_test_prediction(filename, np.array(predictions))
