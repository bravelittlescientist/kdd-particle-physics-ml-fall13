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

class NeuralNetworkClassifier:
    def __init__(self):
        pass

def classify(Xtrain, Ytrain, n_hidden=5):
    """ Use entirety of provided X, Y to predict

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier -- a classifier fitted to Xtrain and Ytrain
    """

    # PyBrain expects data in its DataSet format
    trndata = ClassificationDataSet(Xv.shape[1], 1, nb_classes=2)
    trndata.setField('input', Xtrain)
    trndata.setField('output', Ytrain)
    trndata._convertToOneOfMany() # one output neuron per class

    # build neural net and train it
    net = buildNetwork(trndata.indim, n_hidden, trndata.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    #trainer.trainUntilConvergence()
    trainer.trainEpochs(5)

    # TODO
    # Return a functor that wraps calling predict

    return trainer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: $ python neural_network.py /path/to/data/file/"
    else:
        training = sys.argv[1]
        X,Y,n,f = load_data(training)
        Xt, Xv, Yt, Yv = shuffle_split(X,Y)

        classifier = classify(Xt, Yt)
        #predictions = classifier.predict(Xv)
        tstdata = ClassificationDataSet(Xv.shape[1], 1, nb_classes=2)
        tstdata.setField('input', Xv)
        tstdata.setField('output', Yv)
        tstdata._convertToOneOfMany() # one output neuron per class

        predictions = classifier.testOnClassData(dataset=tstdata)

        print "Decision Tree Accuracy:",acc(Yv, predictions),"%"
