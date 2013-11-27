#!/usr/bin/python2

# Metrics methods as dictated by the KDD competition:
# * ACC
# * AUC
# * Cross-Entropy (log loss)
# * SLQ

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def acc(Yactual, Ypredict):
    """ Returns % correct of prediction

    Arguments:
    Yactual -- Correct Y
    Ypredicted -- Predicted by classifier

    Returns:
    Accuracy score -- % correct of predicted values

    We want to maximize this measure.
    Doc: http://scikit-learn.org/stable/modules/generated/sklearn.metrics
    """
    return accuracy_score(Yactual, Ypredict)

def auc(Yactual, Ypredict):
    """ Area Under Curve for binary classification task.

    Arguments:
    Yactual -- Correct Y
    Ypredicted -- Predicted by classifier

    Returns:
    ROC score -- Area under curve

    We want to maximize this measure
    Doc: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    """
    return roc_auc_score(Yactual, Ypredict)

def cross_entropy(Yactual, Ypredict):
    """ Logistical Loss / Cross-entropy)

    Arguments:
    Yactual -- Correct Y
    Ypredicted -- Predicted by classifier

    Returns:
    Log Loss -- Classification for logistic regression and its friends

    We want to minimize this measure.
    Doc: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """
    return log_loss(Yactual, Ypredict)

def SLQ(Yactual, Ypredict):
    """ Stanford Linear Accelerator Metric.

    Arguments:
    Yactual -- Correct Y
    Ypredicted -- Predicted by classifier

    Returns:
    Log Loss -- Classification for logistic regression and its friends

    TODO.
    """
    pass

def suite(Yactual, Ypredict):
    """ Runs all measures (and SLQ once it's implemented)
    and reports results """
    print "ACC Score",acc(Yactual, Ypredict)
    print "AUC Score",auc(Yactual, Ypredict)
    # print "Logistic Loss",cross_entropy(Yactual, Ypredict)
