#!/usr/bin/python2

# Util contains some methods that will come in handy
# for everything from testing ensemble members to
# deciding which score to submit. Basically a catch-all
# to avoid repeating too much code.

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

def shuffle_split(X,Y):
    """ Basic split and shuffle of X and Y

    Arguments
    X -- Features data
    Y -- Classifications for X

    Returns
    Xt -- data, training
    Xv -- data, validation
    Yt -- classifications, training
    Yv -- classifications, validation
    """
    # Split & shuffle
    shuffle(X,Y)
    Xt, Xv, Yt, Yv = train_test_split(X, Y, train_size=0.75)
    return Xt, Xv, Yt, Yv
