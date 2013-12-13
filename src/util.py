#!/usr/bin/python2

# Util contains some methods that will come in handy
# for everything from testing ensemble members to
# deciding which score to submit. Basically a catch-all
# to avoid repeating too much code.

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from imputation import load_data, impute_missing_data, remove_features_missing_data

def shuffle_split(X,Y, train_size=0.75):
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
    Xt, Xv, Yt, Yv = train_test_split(X, Y, train_size=train_size)
    return Xt, Xv, Yt, Yv

def get_split_training_dataset(train_part=0.75):
    """ Get the phy_train dataset shuffled and split """
    # Impute dataset
    X, Y, n, f = load_data("../data/raw/phy_train.dat")

    # Split and shuffle
    return train_test_split(X, Y, train_size=train_part)

def load_validation_data():
    """ Load training and testing data

    Returns
    Xt -- Imputed training data
    Yt -- training prediction
    Xv -- Imputed validation data
    """
    # Load and impute validation data
    Xv, Yzero, nv, fv = load_data("../data/raw/phy_test.dat", load_y=False)
    Xv = remove_features_missing_data(Xv)

    # Load and impute training data
    Xt, Yt, nt, ft = load_data("../data/raw/phy_train.dat")
    Xt = remove_features_missing_data(Xt)

    return Xt, Yt, Xv

def write_test_prediction(outfile, Ypredict, submission=True):
    f = open(outfile, 'w')
    if submission == True:
        case = 50001
    else:
        case = 0

    for y in Ypredict:
        f.write(str(case) + " " + str(int(y)) + "\n")
        case = case + 1

    f.close()
