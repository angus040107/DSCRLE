from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_scale(x, batch_size, n_nbrs=10):
    '''
    Calculates the scale* based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size

    x:          data for which to compute scale
    batch_size: m in the aforementioned calculation. it is
                also the batch size of spectral net
    n_nbrs:     k in the aforementeiond calculation.

    returns:    the scale*

    *note:      the scale is the variance term of the gaussian
                affinity matrix used by spectral net
    '''
    n = len(x)

    # sample a random batch of size batch_size
    sample = x[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:, n_nbrs - 1])


def squared_distance(X, Y=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    X1 = tf.reshape(X, (1, X.shape[0], X.shape[-1]))
    Y1 = tf.reshape(Y, (Y.shape[0], 1, Y.shape[-1]))
    DXY = tf.reduce_sum(tf.square(X1 - Y1), axis=-1)
    # DXY = tf.norm(X1-Y1, ord='euclidean', axis=-1)

    return DXY


def full_affinity(X, Y=None, scale=1.0):
    DXX = squared_distance(X)
    scale = tf.cast(scale, dtype=tf.float32)
    sigma = tf.constant(scale)
    sigma_squared = tf.pow(sigma, 2)
    sigma_squared = tf.expand_dims(sigma_squared, -1)
    Dx_scaled = DXX / (2 * sigma_squared)
    S = tf.exp(-Dx_scaled)
    return S


def knn_affinity(X, n_nbrs, scale=None, scale_nbr=None, local_scale=None, verbose=False):
    '''
    Calculates the symmetrized Gaussian affinity matrix with k1 nonzero
    affinities for each point, scaled by
    1) a provided scale,
    2) the median distance of the k2th neighbor of each point in X, or
    3) a covariance matrix S where S_ii is the distance of the k2th
    neighbor of each point i, and S_ij = 0 for all i != j
    Here, k1 = n_nbrs, k2=scale_nbr

    X:              input dataset of size n
    n_nbrs:         k1
    scale:          provided scale
    scale_nbr:      k2, used if scale not provided
    local_scale:    if True, then we use the aforementioned option 3),
                    else we use option 2)
    verbose:        extra printouts

    returns:        n x n affinity matrix
    '''
    if isinstance(n_nbrs, np.float):
        n_nbrs = int(n_nbrs)
    elif isinstance(n_nbrs, tf.Variable) and n_nbrs.dtype.as_numpy_dtype != np.int32:
        n_nbrs = tf.cast(n_nbrs, np.int32)
    # get squared distance
    Dx = squared_distance(X)
    # calculate the top k neighbors of minus the distance (so the k closest neighbors)
    nn = tf.nn.top_k(-Dx, n_nbrs, sorted=True)

    vals = nn[0]
    # apply scale
    if scale is None:
        # if scale not provided, use local scale
        if scale_nbr is None:
            scale_nbr = 0
        else:
            print("getAffinity scale_nbr, n_nbrs:", scale_nbr, n_nbrs)
            assert scale_nbr > 0 and scale_nbr <= n_nbrs
        if local_scale:
            scale = -nn[0][:, scale_nbr - 1]
            scale = tf.reshape(scale, [-1, 1])
            scale = tf.tile(scale, [1, n_nbrs])
            scale = tf.reshape(scale, [-1, 1])
            vals = tf.reshape(vals, [-1, 1])
            if verbose:
                vals = tf.Print(vals, [tf.shape(vals), tf.shape(scale)], "vals, scale shape")
            vals = vals / (2 * scale)
            vals = tf.reshape(vals, [-1, n_nbrs])
        else:
            def get_median(scales, m):
                with tf.device('/cpu:0'):
                    scales = tf.nn.top_k(scales, m)[0]
                scale = scales[m - 1]
                return scale, scales

            scales = -vals[:, scale_nbr - 1]
            const = tf.shape(X)[0] // 2
            scale, scales = get_median(scales, const)
            vals = vals / (2 * scale)
    else:
        # otherwise, use provided value for global scale
        vals = vals / (2 * scale ** 2)

    # get the affinity
    affVals = tf.exp(vals)
    # flatten this into a single vector of values to shove in a spare matrix
    affVals = tf.reshape(affVals, [-1])
    # get the matrix of indexes corresponding to each rank with 1 in the first column and k in the kth column
    nnInd = nn[1]
    # get the J index for the sparse matrix
    jj = tf.reshape(nnInd, [-1, 1])
    # the i index is just sequential to the j matrix
    ii = tf.range(tf.shape(nnInd)[0])
    ii = tf.reshape(ii, [-1, 1])
    ii = tf.tile(ii, [1, tf.shape(nnInd)[1]])
    ii = tf.reshape(ii, [-1, 1])
    # concatenate the indices to build the sparse matrix
    indices = tf.concat((ii, jj), axis=1)
    # assemble the sparse Weight matrix
    S = tf.SparseTensor(indices=tf.cast(indices, dtype='int64'), values=affVals,
                        dense_shape=tf.cast(tf.shape(Dx), dtype='int64'))
    # fix the ordering of the indices
    S = tf.sparse.reorder(S)
    # convert to dense tensor
    S = tf.sparse.to_dense(S)
    # symmetrize
    S = (S + tf.transpose(S)) / 2.0;

    return S


def anchor_affinity(X, n_nbrs):
    Dx = squared_distance(X)
    D = -Dx
    [Dsort, Didx] = tf.nn.top_k(D, n_nbrs + 1)
    S = get_S_achor(Dsort, Didx, n_nbrs)
    return S


def sec(X, F, n_nbrs, data_dim, gammad, Wxbf):

    S = anchor_affinity(X, n_nbrs)

    wf1 = tf.linalg.inv(tf.matmul(tf.transpose(X), X) + ((1 - gammad) / gammad) * tf.eye(data_dim))
    wf2 = tf.matmul(tf.transpose(X), F)
    W = tf.matmul(wf1, wf2)
    Wmoment = 0.9 * Wxbf + 0.1 * W
    return S, Wmoment


def kec(X, F, n_nbrs, data_dim, gammad, Wxbf):

    S = knn_affinity(X, n_nbrs)

    wf1 = tf.linalg.inv(tf.matmul(tf.transpose(X), X) + ((1 - gammad) / gammad) * tf.eye(data_dim))
    wf2 = tf.matmul(tf.transpose(X), F)
    W = tf.matmul(wf1, wf2)
    Wmoment = 0.999 * Wxbf + 0.001 * W
    return S, Wmoment


def get_S_achor(Dsort, Didx, neibor=11):
    Dsort = -Dsort
    Xshape = tf.shape(Dsort)
    colidx = Didx[:, 1:neibor]
    colidx = K.expand_dims(colidx, axis=-1)
    rowidx = K.expand_dims(K.expand_dims(tf.range(0, Xshape[0]), axis=-1), axis=-1)
    rowidx = tf.tile(rowidx, [1, tf.shape(colidx)[-2], 1])
    idx = tf.reshape(tf.concat([rowidx, colidx], axis=-1), [-1, 2])

    valuek = -Dsort[:, 1:neibor]
    valuekk = -Dsort[:, neibor]
    valuekk = tf.tile(K.expand_dims(valuekk, axis=-1), [1, tf.shape(valuek)[-1]])

    valuesum = tf.reduce_sum(valuek, axis=-1, keepdims=True)
    valuesum = tf.tile(valuesum, [1, tf.shape(valuek)[-1]])

    valuek = tf.reshape(valuek, [-1])
    valuekk = tf.reshape(valuekk, [-1])
    valuesum = tf.reshape(valuesum, [-1])

    Svalue = (valuekk - valuek) / (valuekk * (neibor - 1) - valuesum + tf.constant(1e-7))

    S = tf.scatter_nd(idx, Svalue, [Xshape[0], Xshape[0]])

    S = (S + tf.transpose(S)) / 2.0;

    return S


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def compute_accuracy(y_true, y_pred):  # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.'''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.'''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
