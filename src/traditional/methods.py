import numpy as np

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph

np.random.seed(0)



def hc_clustering(algo_name, X, cfg):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}
    params = default_base.copy()
    params['n_clusters'] = cfg['n_clusters']

    if(algo_name == 'MiniBatchKMeans'):
        algorithm = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    elif (algo_name == 'AffinityPropagation'):
        algorithm = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])

    elif (algo_name == 'MeanShift'):
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
        algorithm = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    elif (algo_name == 'SpectralClustering'):
        algorithm = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")

    elif (algo_name == 'Ward'):
        connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        algorithm = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)

    elif (algo_name == 'AgglomerativeClustering'):
        connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        algorithm = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)

    elif (algo_name == 'DBSCAN'):
        algorithm = cluster.DBSCAN(eps=params['eps'])

    elif (algo_name == 'OPTICS'):
        algorithm = cluster.OPTICS(min_samples=params['min_samples'],
                                xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])

    elif (algo_name == 'Birch'):
        algorithm = cluster.Birch(n_clusters=params['n_clusters'])

    elif (algo_name == 'GaussianMixture'):
        algorithm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
    else:
        print(1)

    algorithm.fit(X)

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    return y_pred