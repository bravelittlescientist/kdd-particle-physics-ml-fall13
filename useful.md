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

### Evaluation Criteria

* Maximize ACC (correct prediction %). Trivial to compute, # correct / total. [sklearn.metrics.acc](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
* Maximize AUC (area under ROC curve) [ROC Example](http://scikit-learn.org/stable/auto_examples/plot_roc.html#example-plot-roc-py), [sklearn.metrics.roc_auc_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
* Minimize CXE (cross-entropy) [Cross-Entropy / sklearn.metrics.log_loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
* Maximize SLQ (Physics Q-Score). Need to write our own if we want to optimize; good description at [KDD 2004 site](http://osmot.cs.cornell.edu/kddcup/metrics.html).

