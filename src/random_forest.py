#!/usr/bin/python2

# This is a random forest ensemble classifier based on the scikit
# ensembles module.
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.
#
# Decision tree documentation:
# http://scikit-learn.org/stable/modules/tree.html

import sys

from util import get_split_training_dataset
from metrics import suite

import feature_selection_trees as fclassify

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def train(Xtrain, Ytrain, n=350, grid=False):
    """ Use entirety of provided X, Y to train random forest

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier -- A random forest of n estimators, fitted to Xtrain and Ytrain
    """
    if grid == True:
        forest = RandomForestClassifier(max_depth=None, random_state=0, min_samples_split=1,max_features=38)
        parameters = {
            'n_estimators': [200,250,300],
        }

        # Classify over grid of parameters
        classifier = GridSearchCV(forest, parameters)
    else:
        classifier = RandomForestClassifier(n_estimators=n)

    classifier.fit(Xtrain, Ytrain)
    return classifier

if __name__ == "__main__":
    # Let's take our training data and train a random forest
    # on a subset.
    Xt, Xv, Yt, Yv = get_split_training_dataset()
    print "Random Forest Classifier"
    Classifier = train(Xt, Yt)
    suite(Yv, Classifier.predict(Xv))

    # smaller feature set
    Xtimp, features = fclassify.get_important_data_features(Xt, Yt)
    Xvimp = fclassify.compress_data_to_important_features(Xv, features)
    ClassifierImp = train(Xtimp,Yt)
    print "Forest Classiifer, ~25 important features"
    suite(Yv, ClassifierImp.predict(Xvimp))
