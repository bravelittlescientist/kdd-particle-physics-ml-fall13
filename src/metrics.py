#!/usr/bin/python2

# Metrics methods as dictated by the KDD competition:
# * ACC
# * AUC
# * Cross-Entropy (log loss)
# * SLQ

from sklearn.metrics import accuracy_score

def acc(Yactual, Ypredicted):
    """ Returns % correct of prediction """
    return accuracy_score(Yactual, Ypredicted)
