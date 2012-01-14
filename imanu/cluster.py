import numpy as np
from sklearn.cluster.mean_shift_ import MeanShift, estimate_bandwidth
from sklearn.cluster import MiniBatchKMeans


def data_reshape(image):
    """
    Reshape the data in order to perform meanshift

    params:
        image: ndarray, (X, Y, n)

    returns
        ndarray (X*Y, n)
    """
    image_mat = []
    if image.shape[-1] == 3:
        for x, i in enumerate(image):
            for y, j in enumerate(i):
                image_mat.append([x, y, j[0], j[1], j[2]])
    else:
        for x, i in enumerate(image):
            for y, j in enumerate(i):
                image_mat.append([x, y, j])
    return np.array(image_mat)


def meanshift(desc, quantile, hs=16, hr=16, copy=True):
    """
    Do nothing for now...
    """
    if copy:
        desc = desc.copy()
    desc[:, :2] = desc[:, 1:2] / hs
    desc[:, 2:] = desc[:, 1:2] / hr
    bandwidth = estimate_bandwidth(desc, quantile=quantile, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(desc)
    return ms


def k_means(desc, quantile, hs=16, hr=16, copy=True):
    """
    Do nothing for now...
    """
    if copy:
        desc = desc.copy()
    desc[:, :2] = desc[:, 1:2] / hs
    desc[:, 2:] = desc[:, 1:2] / hr

    ms = MiniBatchKMeans(k=quantile) 
    ms.fit(desc)
    return ms
