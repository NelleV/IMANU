"""
============================================
Segmenting Lena into region using mean shift
============================================

"""

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com> 
# License: BSD

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster.mean_shift_ import MeanShift, estimate_bandwidth

lena = sp.misc.lena()
# My computer is crap - I don't have enough ram to compute the clustering on
# the whole lena image. Let's downsample the image by a factor of 4
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

# Lena as an image is useless. Let's create a 512*3 matrix (x, y, value)
lena_mat = []
for x, i in enumerate(lena):
    for y, j in enumerate(i):
        lena_mat.append([x, y, j])

lena_mat = np.array(lena_mat)

quantile_range = np.linspace(0.01, 0.1, 9)
images = []
images.append({"image": lena.copy(), "quantile": "", "clusters": None})

for quantile in quantile_range:
    bandwidth = estimate_bandwidth(lena_mat, quantile=quantile, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(lena_mat)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    lena_clustered = lena.copy()
    lena_clustered_value = lena.copy()
    lena_mat_clustered = lena_mat.copy()
    lena_mat_clustered_value = lena_mat.copy()

    for point, pointb, value in zip(lena_mat_clustered, lena_mat_clustered_value, labels):
        point[2] = value
        pointb[2] = cluster_centers[value, 2]
        lena_clustered[point[0], point[1]] = value
        lena_clustered_value[point[0], point[1]] = cluster_centers[value, 2]

    image = {"image": lena_clustered_value, "quantile": quantile, "clusters":
    n_clusters_}
    images.append(image)

fig = plt.figure()

for i, image in enumerate(images):
    ax = fig.add_subplot(2, 5, i)
    ax.matshow(image['image'])

#fig = plt.figure(1)
#ax = fig.add_subplot(111, projection='3d')
#
#ax.plot(lena_mat[:, 0], lena_mat[:, 1], lena_mat[:, 2], 'w',
#markerfacecolor='#111111', marker='.')
#
#plt.show()
#
# Let's display some of the results






