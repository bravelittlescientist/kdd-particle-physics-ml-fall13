#!/usr/bin/python2

# This is a neural network based on PyBrain,
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.

import sys

from imputation import load_data
from util import shuffle_split
from metrics import acc

from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from numpy import mat

class NeuralNetworkClassifier(object):
    def __init__(self, trainer):
        self.trainer = trainer

    def predict(self, tstdata):
        predictions = self.trainer.testOnClassData(dataset=tstdata)
        '''
        predictions = self.trainer.activateOnDataset(data)
        predictions = predictions.argmax(axis=1)
        #predictions = predictions.reshape(X.shape) ???
        '''
        print predictions # it's giving all 0's!!
        return predictions

def classify(Xtrain, Ytrain, n_hidden=[5], epocs_to_train=5):
    """ Use entirety of provided X, Y to predict

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier -- a classifier fitted to Xtrain and Ytrain
    """

    # Apprently, arrays don't work here as they try to access second dimension size...
    Ytrain = mat(Ytrain).transpose()

    # PyBrain expects data in its DataSet format
    trndata = ClassificationDataSet(Xtrain.shape[1], Ytrain.shape[1], nb_classes=2)
    trndata.setField('input', Xtrain)
    trndata.setField('target', Ytrain)

    trndata._convertToOneOfMany() # one output neuron per class

    # build neural net and train it
    net = buildNetwork(trndata.indim, *(n_hidden + [trndata.outdim]), outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    #trainer.trainUntilConvergence()
    trainer.trainEpochs(epocs_to_train)

    # Return a functor that wraps calling predict
    return NeuralNetworkClassifier(trainer)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        training = '../data/raw/phy_train.dat'
        print "Usage: $ python neural_network.py /path/to/data/file/"
        print "Using default data file:", training
    else:
        training = sys.argv[1]

    X,Y,n,f = load_data(training)
    Xt, Xv, Yt, Yv = shuffle_split(X,Y)

    classifier = classify(Xt, Yt, [100])

    # Apprently, arrays don't work here as they try to access second dimension size...
    Yv = mat(Yv).transpose()

    tstdata = ClassificationDataSet(Xv.shape[1], Yv.shape[1], nb_classes=2)
    tstdata.setField('input', Xv)
    tstdata.setField('target', Yv)
    tstdata._convertToOneOfMany() # one output neuron per class

    predictions = classifier.predict(tstdata)

    print "Decision Tree Accuracy:",acc(Yv, predictions),"%"

