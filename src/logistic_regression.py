#!/usr/bin/python2

# This is a Logistic Classifier based on the
# scikit-learn documentation.

import sys

from imputation import load_data
from util import shuffle_split
from metrics import suite

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def train(Xtrain, Ytrain, C=1e5):
    """ Use entirety of provided X, Y to predict

    Default Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Named Arguments
    C -- regularization parameter

    Returns
    classifier -- a tree fitted to Xtrain and Ytrain
    """
    # Prepare pipeline
    logistic = LogisticRegression()
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    # Grid Search
    n_components = [20, 40, 64]
    Cs = [100, 1000, 1e5]
    estimator = GridSearchCV(pipe,
            dict(pca__n_components=n_components, logistic__C=Cs))
    estimator.fit(Xtrain, Ytrain)

    return estimator

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
        Classifier = train(Xt, Yt)
        print "Pipeline: PCA / Logistic Classifier"
        suite(Yv, Classifier.predict(Xv))
