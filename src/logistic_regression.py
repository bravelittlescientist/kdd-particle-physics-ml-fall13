#!/usr/bin/python2

# This is a Logistic Classifier based on the
# scikit-learn documentation.

import sys

from util import get_split_training_dataset
from metrics import suite

import feature_selection_trees as fclassify

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
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    Classifier = train(Xt, Yt)
    print "PCA/Logistic Classifier"
    suite(Yv, Classifier.predict(Xv))

    # smaller feature set
    Xtimp, features = fclassify.get_important_data_features(Xt, Yt, max_features=10)
    Xvimp = fclassify.compress_data_to_important_features(Xv, features)
    ClassifierImp = train(Xtimp,Yt)
    print "Logistic Classiifer, ~36 important features"
    suite(Yv, ClassifierImp.predict(Xvimp))
