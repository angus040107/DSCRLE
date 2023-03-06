import numpy as np
import cv2
from sklearn import cluster, metrics
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

# from src.sconfig import *
# from src.datatools import *
from sklearn.preprocessing import *

def mkmeans(X, nclusters):
    # X = StandardScaler().fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)
    k_means = cluster.MiniBatchKMeans(n_clusters=nclusters)
    k_means.fit(X)
    y_pred = k_means.predict(X)
    return y_pred

def msc(X, params):
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    spectral.fit(X)
    y_pred = spectral.labels_.astype(int)
    return y_pred

def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose( metrics.confusion_matrix(y_true, y_pred) )
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)


def cluster_kappa(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    pe_rows = np.sum(cm, axis=0)
    pe_cols = np.sum(cm, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)

    w = np.transpose(cm)
    row_ind, col_ind = linear_assignment(w.max() - w)
    sum_right = w[row_ind, col_ind].sum()
    po = sum_right / float(sum_total)

    return (po - pe) / (1 - pe)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose( metrics.confusion_matrix(y_true, y_pred) )
    row_ind, col_ind = linear_assignment(w.max() - w)
    return np.sum([w[row_ind[i], col_ind[i]] for i in range(0,len(row_ind))]) * 1.0 / y_pred.size

def get_results(y_preds, y_true, cfg):
    results = []
    if cfg['use_kmeans']:
        y_preds_kmeans = mkmeans(y_preds, cfg['n_clusters'])
        # y_preds_kmeans = y_preds.argmax(1)
    else:
        y_preds_kmeans = y_preds
    results.append(cluster_acc(y_true, y_preds_kmeans))
    results.append(cluster_purity(y_true, y_preds_kmeans))
    results.append(metrics.rand_score(y_true, y_preds_kmeans))
    # results.append(metrics.adjusted_rand_score(y_true, y_preds_kmeans))
    results.append(metrics.normalized_mutual_info_score(y_true, y_preds_kmeans))
    results.append(cluster_kappa(y_true, y_preds_kmeans))
    results.append(metrics.v_measure_score(y_true, y_preds_kmeans))
    return results, y_preds_kmeans

def make_batches(size, batch_size):
    '''
    generates a list of (start_idx, end_idx) tuples for batching data
    of the given size and batch_size

    size:       size of the data to create batches for
    batch_size: batch size

    returns:    list of tuples of indices for data
    '''
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

def ineedtosaveresults(name, data=None, filepath='result.txt', moshi='a'):
    f = open(filepath, moshi)
    if(data == None):
        f.write(name)
        f.write('\n')
    elif(type(data) == 'dict'):
        for key, value in data.items():
            f.write(key + ':' + str(value))
            f.write('\n')
    elif(type(data) == 'list'):
        f.write(name + ':' + str(data))
        f.write('\n')
    else:
        f.write(name + ':' + str(data))
        f.write('\n')
    f.close()

def label2img(inpredict, inlabel):

    label_flatten = inlabel.flatten()
    if(len(inpredict) == len(label_flatten)):
        predictions = label_flatten
    else:
        predictions = np.zeros(np.shape(label_flatten))
        predictions[np.nonzero(label_flatten)] = inpredict

    predictions = predictions.reshape(np.shape(inlabel))

    if(len(np.shape(inlabel)) > 1):
        label_arr = cv2.cvtColor(np.uint8(inlabel / inlabel.max() * 255.0), cv2.COLOR_GRAY2RGB)
        label_arr = cv2.applyColorMap(label_arr, cv2.COLORMAP_JET)

        predictions_arr = cv2.cvtColor(np.uint8(predictions / predictions.max() * 255.0), cv2.COLOR_GRAY2RGB)
        predictions_arr = cv2.applyColorMap(predictions_arr, cv2.COLORMAP_JET)
    else:
        label_arr = inlabel
        predictions_arr = predictions

    return label_arr, predictions_arr

def imgsave(label_arr, predictions_arr, save_path, metheds):
    cv2.imwrite(save_path + '/' + metheds +'_label_arr.png', label_arr)
    cv2.imwrite(save_path + '/' + metheds +'_predictions_arr.png', predictions_arr)
    return True

def resultshow(label_arr, predictions_arr):
    cv2.imshow('label_arr', label_arr)
    cv2.imshow('predictions_arr', predictions_arr)
    return True

#
# cfg = returncfg()
# data, label, label_ = get_data('indiapines')
# cfg['n_clusters'] = len( np.unique(label) )
#
# minmaxscaler = MinMaxScaler()
# data = minmaxscaler.fit_transform(data)
# # data = StandardScaler().fit_transform(data)
# metric_list = ['acc','purity','rand_score','adjusted_rand_score','normalized_mutual_info_score','adjusted_mutual_info_score','v_measure_score']
# results, y_preds_kmeans = get_results(data, label, cfg)
# # label_arr, predictions_arr = label2img(results, label)
# # resultshow(label_arr, predictions_arr)
# print(metric_list)
# print(results)