import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose( confusion_matrix(y_true, y_pred) )
    row_ind, col_ind = linear_assignment(w.max() - w)
    return np.sum([w[row_ind[i], col_ind[i]] for i in range(0,len(row_ind))]) * 1.0 / y_pred.size