
import numpy as np
import scipy.fftpack

from tslearn.clustering import TimeSeriesKMeans, KShape
import itertools


def get_KMeans_label(data, CC, t):
    ## label cluster based on smaller euclidean distance to both CC
    n_clusters = CC.shape[0]
    dist2CC = np.zeros((data.shape[0], n_clusters, 2))
    for i in range(n_clusters):
        dist_euclidean = np.corrcoef(CC[i], data)[0, 1:]
        dist2CC[:, i, 0] = np.sqrt(np.sum(np.subtract(data, CC[i]) ** 2, axis=1))
        dist2CC[:, i, 1] = np.corrcoef(CC[i], data)[0, 1:]
    # label based on smallest euclidean distance to different cluster centers
    if t == 'mean':
        y = np.argmin(dist2CC[:, :, 0], axis=1)
    else:
        y = np.argmax(dist2CC[:, :, 1], axis=1)
    return y


def update_cluster(data, CC0, t):
    ## input:
    # CC0: current cluster centers (centroids)
    # data: data to cluster

    ## output
    # CC: new cluster centers, y: labels, dist: total squared dist

    ## 1. get current label
    y = get_KMeans_label(data, CC0, t)
    ## 2. calculate new cluster centers based on new labels (y)
    # CC = np.zeros((CC0.shape[0], data.shape[1]))
    CC = np.copy(CC0)
    for i in np.unique(y):
        CC[i] = np.mean(data[y == i], axis=0)  # Calculate centroids as mean of the cluster

    ## 3. update label
    y = get_KMeans_label(data, CC, t)
    ## 4. sum of squared distances of each electrode from closest CC
    dist = 0
    for i in np.unique(y):
        dist += np.sum(np.sum(np.subtract(data[y == i], CC[i]) ** 2, axis=1))

    return CC, y, dist


def KMeans_similarity(data, n_clusters, t='mean', max_it=50):
    # data = data.T  # transpose data
    idx = np.random.choice(data.shape[0], n_clusters)  # Step 2 => randomly select 2 points
    CC_init = data[idx]

    CC0, y, dist = update_cluster(data, CC_init, t)  # First loop started from random points
    for i in range(max_it):
        CC, y, dist = update_cluster(data, CC0, t)  # Then from centroids previous clustering
        if np.array_equal(np.around(CC, 3), np.around(CC0, 3)):
            break
        else:
            CC0 = CC
    return dist, y, CC


def ts_cluster(X, n=2, method='euclidean'):
    # methods: 'dtw', 'euclidean', 'shape'

    if method == 'shape':  # cross-correlation based
        X = np.expand_dims(X, -1)
        ks = KShape(n_clusters=n, n_init=1, random_state=0).fit(X)
        cc = ks.cluster_centers_  # centroids
        y = ks.predict(X)  # cluster label
        CC = cc[:, :, 0]
    elif method == 'similarity':
        dist, y, CC = KMeans_similarity(X, 2, t=method, max_it=50)
    else:
        X = np.expand_dims(X, -1)
        ks = TimeSeriesKMeans(n_clusters=n, metric=method, max_iter=10, max_iter_barycenter=10, random_state=0).fit(X)
        cc = ks.cluster_centers_  # centroids
        y = ks.predict(X)  # cluster label
        CC = cc[:, :, 0]
    return CC, y


def dba_cluster(X, n=2, method='dtw'):
    X = np.expand_dims(X, -1)
    km_dba = TimeSeriesKMeans(n_clusters=n, metric=method, max_iter=10, max_iter_barycenter=10, random_state=0).fit(X)
    cc = km_dba.cluster_centers_
    y = km_dba.predict(X)
    dist = km_dba.transform(X)
    dist_cc = np.max(km_dba.transform(km_dba.cluster_centers_))
    return cc[:, :, 0], y, dist, dist_cc


def get_cluster_pred(sc, rc, LL_CCEP, EEG_resp, Fs=500):
    lists = LL_CCEP[
        ~np.isnan(LL_CCEP.zLL.values) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['Condition'] > 0)]
    conds_trials = lists.Condition.values.astype('int')
    stimNum_all = lists.Num.values.astype('int')
    d = 0.5
    trials = EEG_resp[rc, stimNum_all, :]
    trials_z = scipy.stats.zscore(trials, 1)
    data = trials_z[:, np.int64(1 * Fs):np.int64((1 + d) * Fs)]
    cc, y_pred, dist = dba_cluster(np.expand_dims(data, -1))
    D = dist / np.max(dist)

    pred_loss = np.zeros((2,))
    i = 0
    for cond in np.unique(conds_trials):
        d = 0
        for x, y in itertools.combinations(y_pred[conds_trials == cond], 2):
            d += np.square(x - y)
        pred_loss[i] = d  # np.sqrt(d)
        i = 1 + i
    return pred_loss
