#!/usr/bin/python2

# This is an extra random trees ensemble classifier based on the scikit
# ensembles module.

import sys
import numpy as np

from util import get_split_training_dataset

from sklearn.ensemble import ExtraTreesClassifier

def get_important_features(Xtrain, Ytrain, n=250, threshold=0.01, verbose=False):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Optional Arguments
    n -- number of ensemble members
    threshold -- threshold of importance above which a feature is relevant
    verbose -- if true, prints results of ranking

    Returns
    ranking -- a ranked list of indices of important features
    """
    # Train and fit tree classifier ensemble
    classifier = ExtraTreesClassifier(n_estimators=n, random_state=0)
    classifier.fit(Xtrain, Ytrain)

    # Compute important features
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    ranking = [[indices[f], importances[indices[f]]] for f in range(Xtrain.shape[1])]
    ranking = filter(lambda r: r[1] >= threshold, ranking)

    if verbose:
        for r in range(len(ranking)):
            print str(r+1) + ". ", ranking[r][0], ranking[r][1]

    return ranking

def compress_data_to_important_features(X, important):
    """ Removes all but important features from X

    Arguments
    X -- numpy array with n samples and d features
    important -- a list of features to keep

    Returns
    Ximportant -- X containing only important features
    """
    remove = filter(lambda c: c not in important, range(X.shape[1]))
    Ximportant = np.delete(X, remove, 1)

    return Ximportant

def get_important_data_features(X, Y, max_features=39, threshold=.01):
    """ Returns X with only important feature columns included (and their indices)

    Arguments
    X -- a np array with n samples and d features
    Y -- a np array with results

    Optional arguments
    threshold -- a threshold above which a feature is important, default is .01
    max_features -- maximum features, default is 39 (half of original data)

    Returns
    Ximportant -- important features of X
    indices -- where to find numeric features, for use on test data
    """
    # Get important features
    ranking = get_important_features(X, Y, threshold=threshold, verbose=True)

    # If there are more than the max allowed, trim them
    # Won't be effected if there are fewer.
    ranking = ranking[:max_features]

    # Compress dataset
    indices = [r[0] for r in ranking]
    Ximportant = compress_data_to_important_features(X, indices)

    return Ximportant, indices

if __name__ == "__main__":
    # Let's take our training data and use a forest
    # to select the best features...
    # First, split the data into a training and test set
    Xt, Xv, Yt, Yv = get_split_training_dataset()

    # Let's just get the top 10 features...
    Ximp, features = get_important_data_features(Xt, Yt, max_features=80)

    # Do it for test data too...
    #Xvimp = compress_data_to_important_features(Xv, features)

    #print "Original training data:",Xt.shape[0],"samples,",Xt.shape[1],"features"
    #print "Original validation data:",Xv.shape[0],"samples,",Xv.shape[1],"features"
    #print "Important training data:",Ximp.shape[0],"samples,",Ximp.shape[1],"features"
    #print "Important validation data:",Xvimp.shape[0],"samples,",Xvimp.shape[1],"features"
