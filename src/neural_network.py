#!/usr/bin/python2

# This is a neural network based on PyBrain,
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.

import sys

from imputation import load_data, remove_features_missing_data
from util import shuffle_split
from metrics import acc
from feature_selection_trees import *

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from numpy import mat

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def convert_to_pybrain_dataset(X, Y=None):
    if Y is None:
        Y = [0]*X.shape[0]
        
    # Apprently, arrays don't work here as they try to access second dimension size...
    Y = mat(Y).transpose()

    data = ClassificationDataSet(X.shape[1], Y.shape[1], nb_classes=2)
    data.setField('input', X)
    data.setField('target', Y)
    data._convertToOneOfMany() # one output neuron per class
    return data

class NeuralNetworkClassifier(object):
    def __init__(self, n_hidden=[5], epochs_to_train=5):
        self.trainer = None
        self.Xt = None
        self.Yt = None
        
        self.params = {}
        self.params['n_hidden'] = n_hidden
        self.params['epochs_to_train'] = epochs_to_train

    def predict(self, tstdata):
        predictions = self.trainer.testOnClassData(dataset=convert_to_pybrain_dataset(tstdata))
        return predictions

    def get_params(self, deep=False):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self
    
    def score(self, X, y):
        """Returns a score of how well the classifier is trained to predict the data."""
        return acc(y, self.trainer.testOnClassData(dataset=convert_to_pybrain_dataset(X,y)))

    def fit(self, Xtrain, Ytrain):
        """ Use entirety of provided X, Y to predict

        Arguments
        Xtrain -- Training data
        Ytrain -- Training prediction
        n_hidden -- each entry in the list n_hidden tells how many hidden nodes at that layer
        epocs_to_train -- number of iterations to train the NN for

        Returns
        classifier -- a classifier fitted to Xtrain and Ytrain
        """
        
        self.Xt = Xtrain
        self.Yt = Ytrain
        n_hidden = self.params['n_hidden']
        epochs_to_train = self.params['epochs_to_train']

        # PyBrain expects data in its DataSet format
        trndata = convert_to_pybrain_dataset(Xtrain, Ytrain)

        # build neural net and train it
        net = buildNetwork(trndata.indim, *(n_hidden + [trndata.outdim]), outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

        with open('nn_progress_report.txt', 'a') as f:
            f.write('training %s for %d epochs\n' % (self.params, epochs_to_train))

        #trainer.trainUntilConvergence()
        trainer.trainEpochs(epochs_to_train)

        # Return a functor that wraps calling predict
        self.trainer = trainer

def test_accuracy(classifier, Xt, Yt, Xv, Yv):
    # Apprently, arrays don't work here as they try to access second dimension size...
    Yv = mat(Yv).transpose()
    Yt = mat(Yt).transpose()

    trndata = convert_to_pybrain_dataset(Xy, Yt)
    # check accuracy of predictions on test data
    tstdata = convert_to_pybrain_dataset(Xv, Yv)

    predictions = classifier.predict(trndata)
    print "Neural Net Train Accuracy:",acc(Yt, predictions),"%"
    predictions = classifier.predict(tstdata)
    print "Neural Net Test Accuracy:",acc(Yv, predictions),"%"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        training = '../data/raw/phy_train.dat'
        print "Usage: $ python neural_network.py /path/to/data/file/"
        print "Using default data file:", training
    else:
        training = sys.argv[1]

    impute_data = True
    # load data from file, imputing data and/or removing some features if requested,
    # then shuffle and split into test and validation
    X,Y,n,f = load_data(training)
    if impute_data:
        X = remove_features_missing_data(X)
    Xt, Xv, Yt, Yv = shuffle_split(X,Y)

    # Let's just get the top 10 features...
    Xt, features = get_important_data_features(Xt, Yt, max_features=25)
    # Do it for test data too...
    Xv = compress_data_to_important_features(Xv, features)

    classifier = NeuralNetworkClassifier()

    param_space = [{'n_hidden' : [[10], [25], [50], [100], [200], [500]], 'epochs_to_train' : [50, 150]}, # explore n_hidden
              {'n_hidden' : [[50], [100]], 'epochs_to_train' : [10, 50, 100, 200, 500]}, # explore # epochs
              {'n_hidden' : [[25, 50, 25], [50, 50], [100, 75, 50, 25], [25, 50, 75, 100], [25, 500, 100, 200, 150, 50]], 'epochs_to_train' : [50, 150]}, # explore multiple hidden layers
              ]
    param_search = GridSearchCV(classifier, param_space, n_jobs=8)
    param_search.fit(Xt, Yt)
    print param_search.grid_scores_ # print scores for each set of parameters
    print classification_report(Yv, param_search.predict(Xv))

    #test_accuracy(classifier, Xt, Yt, Xv, Yv)
