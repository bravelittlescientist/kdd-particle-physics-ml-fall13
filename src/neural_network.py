#!/usr/bin/python2

# This is a neural network based on PyBrain,
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.

import sys, os

from imputation import load_data, remove_features_missing_data
from util import shuffle_split, load_validation_data, write_test_prediction
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

    def predict(self, input_data):
        input_data = convert_to_pybrain_dataset(input_data)

        predictions = self.trainer.testOnClassData(dataset=input_data)
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

    def save_test_results(self, Xt, filename="nn_predictions.txt"):
        predictions = self.predict(Xt)
        write_test_prediction(filename, np.array(predictions))
        

def test_accuracy(classifier, Xt, Yt, Xv, Yv, filename=None):
    # Apprently, arrays don't work here as they try to access second dimension size...
    Yv = mat(Yv).transpose()
    Yt = mat(Yt).transpose()

    predictions = classifier.predict(Xt)
    print "Neural Net Train Accuracy:",acc(Yt, predictions),"%"
    predictions = classifier.predict(Xv)
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
    X, Y, Xv = load_validation_data()
    if impute_data:
        X = remove_features_missing_data(X)
    Xt, Xv, Yt, Yv = shuffle_split(X,Y)

    # get the top features, running in parallel
    children = []
    for n_features in [23, 21, 19, 17]:
        children.append(os.fork())
        if children[-1]:
            continue
        Xt, features = get_important_data_features(Xt, Yt, max_features=n_features)
        # Do it for test data too...
        Xv = compress_data_to_important_features(Xv, features)

        if False: #for running a NN with specific parameters and outputting predictions on the test data
            classifier = NeuralNetworkClassifier(n_hidden=[20], epochs_to_train=1)
            classifier.fit(Xt, Yt)

            # test the trained classifier
            test_accuracy(classifier, Xt, Yt, Xv, Yv)
            classifier.save_test_results(Xv)

        else:
            classifier = NeuralNetworkClassifier()

            param_spaces = [
                [{'n_hidden' : [[10], [25], [50], [100], [200], [500]], 'epochs_to_train' : [50, 150]}, # explore n_hidden
                 {'n_hidden' : [[50], [100]], 'epochs_to_train' : [10, 50, 100, 200, 500]}, # explore # epochs
                 {'n_hidden' : [[25, 50, 25], [50, 50], [100, 75, 50, 25], [25, 50, 75, 100], [25, 500, 100, 200, 150, 50]], 'epochs_to_train' : [50, 150]}, # explore multiple hidden layers
                 ],
                [{'n_hidden' : [[50, 25, 10]], 'epochs_to_train' : [500]}, # explore multiple hidden layers a bit further
                 {'n_hidden' : [[100], [200]], 'epochs_to_train' : [1000]}, # explore larger # epochs
                 {'n_hidden' : [[150], [125], [175]], 'epochs_to_train' : [200]}, # explore the order 100 n_hidden range
                 {'n_hidden' : [[5], [8], [10], [15], [20], [70], [85]], 'epochs_to_train' : [100]}, # explore the order 10 n_hidden range
                 ],
                [{'n_hidden' : [[5], [10], [20]], 'epochs_to_train' : [50, 100, 150]}, # explore # epochs for some of the better n_hidden values
                 ],
                [{'n_hidden' : [[20]], 'epochs_to_train' : [150, 250, 400]}, # explore # epochs for some of the better n_hidden values
                 ],
                [{'n_hidden' : [[5],[20]], 'epochs_to_train' : [150]},
                 ], #4
                [{'n_hidden' : [[1]], 'epochs_to_train' : [1]}, # FOR TESTING
                 ],
                ]

            param_space = param_spaces[-1]

            param_search = GridSearchCV(classifier, param_space, n_jobs=2)
            param_search.fit(Xt, Yt)

            with open('nn_results_%dfeatures.txt' % n_features, 'w') as f:
                f.write("%d features:\n%s" % (n_features, str(param_search.grid_scores_).replace(',}','}\n'))) # print scores for each set of parameters
                f.write(classification_report(Yv, param_search.predict(Xv)))
            
    if children[-1] != 0:
        for pid in children:
            os.wait()
