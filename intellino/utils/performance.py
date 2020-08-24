import numpy as np


def measure_performance(confusion_matrix):
    """
    Calculate TP, TN, FN, FP for each classes
    """
    # https://neosla.tistory.com/18

    axis, _ = confusion_matrix.shape

    TP, TN, FN, FP = [np.zeros((axis)) for i in range(4)]
    for i in range(axis):
        TP[i] = np.diag(confusion_matrix)[i]
        TN[i] = np.sum(np.diag(confusion_matrix)) - TP[i]
        FN[i] = np.sum(confusion_matrix[i][:]) - TP[i]
        FP[i] = np.sum(confusion_matrix) - (TP[i] + TN[i] + FN[i])

    return TP, TN, FN, FP


def accuracy(TP, TN, FN, FP):
    return ((TP + TN) / (TP + TN + FP + FN))


def recall(TP, FN):
    return (TP / (TP + FN))


def precision(TP, FP):
    return (TP / (TP + FP))