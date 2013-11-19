# Useful Scikit-Learn Operations

Importing library

    >>> import sklearn

### Basics - Shuffling and Splitting

Some Starter Data

    >>> X = [[x x*x] for x in range(1,11)]
    [[1, 1],
     [2, 4],
     [3, 9],
     [4, 16],
     [5, 25],
     [6, 36],
     [7, 49],
     [8, 64],
     [9, 81],
     [10, 100]]

    >>> Y = [x*x*x for x in range(1,11)]
    [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]

Shuffling Related Data - Basic

    >>> sklearn.utils.shuffle(X, Y)
    [array([[  4,  16],
           [ 10, 100],
           [  1,   1],
           [  2,   4],
           [  8,  64],
           [  5,  25],
           [  3,   9],
           [  7,  49],
           [  6,  36],
           [  9,  81]]),
    array([  64, 1000,    1,    8,  512,  125,   27,  343,  216,  729])]

Splitting Related Data - Basic

    >>> from sklearn import cross_validation
    >>> x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(X, Y, train_size = 0.75)
    
    >>> [x_train, x_validation, y_train, y_validation]
    [array([[ 10, 100],
           [  1,   1],
           [  5,  25],
           [  6,  36],
           [  9,  81],
           [  8,  64],
           [  3,   9]]),
    array([[ 4, 16],
          [ 2,  4],
          [ 7, 49]]),
    array([1000,    1,  125,  216,  729,  512,   27]),
    array([ 64,   8, 343])]

### Loading Datasets

### Algorithms
