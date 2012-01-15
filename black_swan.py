import numpy as np
from scipy.misc import imsave

from sklearn.externals.joblib import Memory
from skimage import color

from imanu import data
from imanu import descriptors
from imanu import cluster

mem = Memory(cachedir='.')
image = data.black_swan()

for k in [0.005, 0.05, 0.01, 0.1]:
    for hs in [1]:
        for hr in [100, 1000]:
            im = image.copy()
            print 'computing for cluster %d' % k
            print 'RGB'
            im = image.copy()
            desc = mem.cache(descriptors.extract_color_descriptors)(im)
            descs = desc.astype(float)
            ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
            labels = ms.labels_
            num = np.sqrt(len(labels))
            results = labels.reshape((num, num)).T
            results = ms.cluster_centers_[results][:, :, :-2]

            imsave('images/black_swan_color_rgb_meanshift_%d_%d_%d.png' % (k * 1000,
                                                                     hs * 100,
                                                                     hr),
                   results)

            print 'XYZ'
            im = image.copy()

            desc = mem.cache(descriptors.extract_color_descriptors)(
                                    im,
                                    space='xyz')
            descs = desc.astype(float)
            ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
            labels = ms.labels_
            num = np.sqrt(len(labels))
            results = labels.reshape((num, num)).T
            results = ms.cluster_centers_[results][:, :, :-2]

            imsave('images/black_swan_color_xyz_meanshift_%d_%d_%d.png' % (k *
            1000,
                                                                     hs * 100,
                                                                     hr),
                   color.xyz2rgb(results))

            print 'HSV'
            im = image.copy()

            desc = mem.cache(descriptors.extract_color_descriptors)(
                        im,
                        space='hsv')
            descs = desc.astype(float)
            ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
            labels = ms.labels_
            num = np.sqrt(len(labels))
            results = labels.reshape((num, num)).T
            results = ms.cluster_centers_[results][:, :, :-2]

            imsave('images/black_swan_color_hsv_meanshift_%d_%d_%d.png' % (
                                                                     k * 1000,
                                                                     hs * 100,
                                                                     hr),
                   color.hsv2rgb(results))
#
#            print 'GRAY'
#            im = image.copy()
#
#            desc = mem.cache(descriptors.extract_color_descriptors)(
#                        im,
#                        space='gray')
#            descs = desc.astype(float)
#            ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
#            labels = ms.labels_
#            num = np.sqrt(len(labels))
#            results = labels.reshape((num, num)).T
#            results = ms.cluster_centers_[results][:, :, :-2].reshape((num,
#                                                                       num))
#
#            imsave('images/black_swan_color_gray_meanshift_%d_%d_%d.png' % (k * 100,
#                                                                      hs,
#                                                                      hr),
#                    results)
