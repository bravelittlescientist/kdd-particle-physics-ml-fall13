#!/usr/bin/python2

# This is an ensemble of good classifiers, meant to vote on some set of
# training data.
# Random Forest
# Extra Random Forest
# Logistic Regression
# AdaBoost
# GradientBoost
# Majority vote wins.

import sys

import adaboost
import extra_randomized_trees
import gradient_boost
import random_forest
import logistic_regression

from util import write_test_prediction, load_validation_data
from metrics import acc

import numpy as np

if __name__ == "__main__":
    # First obtain our training and testing data
    Xt, Yt, Xv = load_validation_data()

    # Now, we train each classifier on the training data
    classifiers = [
        adaboost.train(Xt, Yt),
        extra_randomized_trees.train(Xt, Yt),
        gradient_boost.train(Xt, Yt),
        random_forest.train(Xt, Yt),
        logistic_regression.train(Xt, Yt),
        ]
    
    # Train another classifier on the ensembles output training predictions
    # for each sample in the training data
    training_predictions = np.mat([[c.predict(sample)[0] for c in classifiers] for sample in Xt])

    meta_classifier = logistic_regression.train(training_predictions, Yt)

    # Check results on training data
    print "Accuracy for individual classifiers:", [acc(Yt, c.predict(Xt)) for c in classifiers]
    predictions = np.mat([c.predict(Xt) for c in classifiers]).transpose()
    print "Accuracy for ensemble classifier:", acc(Yt, meta_classifier.predict(predictions))

    ### TEST DATA ###

    # Predict on test data using the ensemble and meta classifier
    predictions = np.mat([c.predict(Xv) for c in classifiers]).transpose()
    final_predictions = meta_classifier.predict(predictions)

    # Final accuracy
    write_test_prediction("ensemble_predictions.txt", np.array(final_predictions))
