#!/usr/bin/python2

# This is an optimized version of gradient boost.

import sys

import gradient_boost
import feature_selection_trees as fclassify

from util import write_test_prediction, load_validation_data
from metrics import acc

import numpy as np

if __name__ == "__main__":
    # First obtain our training and testing data
    Xt, Yt, Xv = load_validation_data()

    # Train a gradietn boost classifier on it.
    gboost = gradient_boost.train(Xt, Yt)
    Yhat = gboost.predict(Xv)

    # Final accuracy
    write_test_prediction("gboost_optimal_2.txt", Yhat)
