#!/usr/bin/python2

# This is a neural network based on PyBrain,
# taking as input an X and Y and any tree complexity parameters, and
# returning a classifier that can then be analyzed with the classifier.
# See the example in the main method for that and error-checking.

import sys

from util import write_test_prediction, load_validation_data
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

    def predict(self, data):
        predictions = testOnClassData(dataset=tstdata)
        '''
        predictions = self.trainer.activateOnDataset(data)
        predictions = predictions.argmax(axis=1)
        #predictions = predictions.reshape(X.shape) ???
        '''
        return predictions

def classify(Xtrain, Ytrain, n_hidden=5):
    """ Use entirety of provided X, Y to predict

    Arguments
    Xtrain -- Training data
    Ytrain -- Training prediction

    Returns
    classifier -- a classifier fitted to Xtrain and Ytrain
    """

    # PyBrain expects data in its DataSet format
    trndata = ClassificationDataSet(Xtrain.shape[1], nb_classes=2)
    trndata.setField('input', Xtrain)
    # Apprently, arrays don't work here as they try to access second dimension size...
    trndata.setField('target', mat(Ytrain).transpose())

    trndata._convertToOneOfMany() # one output neuron per class

    # build neural net and train it
    net = buildNetwork(trndata.indim, n_hidden, trndata.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    trainer.trainUntilConvergence()
    #trainer.trainEpochs(5)

    print "trained"
    #trainer.trainEpochs(5)

    # Return a functor that wraps calling predict
    return NeuralNetworkClassifier(trainer)

if __name__ == "__main__":
    # First obtain our training and testing data
    # Training has 50K samples, Testing 100K
    Xt, Yt, Xv = load_validation_data()

    # Run Neural Network over training data
    classifier = classify(Xt, Yt)

    # Prepare validation data and predict
    tstdata = ClassificationDataSet(Xv.shape[1], 1, nb_classes=2)
    tstdata.setField('input', Xv)
    tstdata._convertToOneOfMany() # one output neuron per class

    predictions = classifier.predict(tstdata)

    # Write prediction to file
    write_test_prediction("out_nn.txt", np.array(majority))
