#!/usr/bin/python2

# Inspired by scikit-learn's imputation example, found here:
# http://scikit-learn.org/stable/auto_examples/imputation.html
#
# ... and their preprocessing documentation, found here:
# http://scikit-learn.org/stable/modules/preprocessing.html
#
# This file provides functions to handle the preprocessing of the 8 features
# in our dataset with missing data (represented as 999.0 or 9999.0). It also
# includes a function for removing the 8 features missing data. It may be
# useful to comment on the performance comparison of these in our report as a
# basis for selecting an imputation strategy.

import sys

import numpy as np
import numpy.ma as ma

from sklearn.preprocessing import Imputer

def load_data(data_path, load_y=True):
    """ Loads training dataset with numpy.

    Arguments:
    data_path -- physics data formatted .dat file

    Returns:
    X -- physics attribute values
    Y -- physics target classifications
    n_samples -- size of dataset
    n_features -- number of features (expect 78)
    """
    X = np.loadtxt(data_path, usecols=range(2,80))
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if load_y == True:
        Y = np.loadtxt(data_path, usecols=(1,))
    else:
        Y = np.array([0 for n in range(n_samples)])

    return X,Y,n_samples,n_features

def remove_features_missing_data(X):
    """ Removes all datapoints that contain missing values.

    Arguments:
    datapoints -- numpy data array containing missing values.

    Returns:
    X_reduced -- a numpy data array without features missing values

    We will used the masked arrays module of numpy.
    """
    # Mask 999.0
    X_reduced = ma.masked_values(X, 999.0)

    # Mask 9999.0
    X_reduced = ma.masked_values(X_reduced, 9999.0)

    # Compress columns containing the masked values
    X_reduced = np.ma.compress_cols(X_reduced)

    return X_reduced

def impute_missing_data(datapoints, strategy='mean'):
    """ Inputes values for the 8 features missing data

    Arguments:
    datapoints -- X, a dataset with missing values represented 999.0 and 9999.0
    strategy [optional] -- an imputation strategy,
        e.g., mean, median, or most_frequent

    Returns:
    X_imputed -- a dataset with missing values imputed according to the
        provided or default (mean) strategy.

    Uses the scikit-learn Imputer class.
    """
    # First we will replace our placeholder values with NaN to only have
    # to run one imputation.
    np.putmask(datapoints, datapoints == 999.0, np.NaN)
    np.putmask(datapoints, datapoints == 9999.0, np.NaN)

    # Now create an imputer over NaN values, and average over axis=0 (columns)
    # Then, fit the imputer to the dataset.
    imp = Imputer(strategy=strategy, axis=0)
    X_imputed = imp.fit_transform(datapoints)

    return X_imputed

if __name__ == "__main__":
    """ Load and impute training data """

    # Let's load raw/phy_train.dat, our physics training data.
    # load_data will provide data and targets, as well as
    # quantities of data and features, respectively.
    X,Y,n_samples,n_features = load_data(sys.argv[1])

    # Before we impute, let's get the data with the features
    # affected by missing data removed completely. If we remove datapoints
    # with missing data, we will only have just over 6000, so that operation
    # is irrelevant to us.
    X_compressed = remove_features_missing_data(X)

    # X consists of 78 features, 8 of which contain 999.0 or 9999.0 to
    # represent missing values. For the purposes of this example, we want to
    # impute them rather than ignoring them.
    X_imputed = impute_missing_data(X)
