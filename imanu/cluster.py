import numpy as np
from sklearn.cluster.mean_shift_ import MeanShift, estimate_bandwidth

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


def meanshift_color(image, quantile, hs=16, hr=16):
    """
    Do nothing for now...
    """
    image_mat = data_reshape(image)

    image_mat[:, :2] = image_mat[:, 1:2] / hs
    image_mat[:, 2:] = image_mat[:, 1:2] / hr
    bandwidth = estimate_bandwidth(image_mat, quantile=quantile, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(image_mat)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    image_clustered = image.copy()

    for point, value in zip(image_mat, labels):
        point[2] = value
        image_clustered[point[0], point[1]] = image[cluster_centers[value, 0],
                                                    cluster_centers[value, 1],
                                                    :]

    image = {"image": image_clustered,
             "quantile": quantile,
             "clusters": n_clusters_}



