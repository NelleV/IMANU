import numpy as np
from scipy.misc import imsave

from sklearn.externals.joblib import Memory

from imanu import data
from imanu import descriptors
from imanu import cluster

mem = Memory(cachedir='.')
image = data.lena(small=False, gray=False)

for k in [0.005, 0.05, 0.01, 0.1, 0.5]:
    for hs in range(1, 20, 10):
        for hr in range(1, 20, 10):
            im = image.copy()
            print 'computing for cluster %d' % k
            print 'texture'
            for num_bins in range(16, 128, 40):
                desc = mem.cache(descriptors.extract_descriptors)(
                                        im,
                                        use_hog=False,
                                        verbose=False, num_bins=num_bins)
                descs = desc.astype(float)
                #descs[:, :-5] = normalize(descs[:, :-5])
                #descs[:, -5:-2] = normalize(descs[:, -5:-2])
                ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
                labels = ms.labels_
                num = np.sqrt(len(labels))
                results = labels.reshape((num, num)).T

                imsave('images/lena_texture%d_meanshift_%d_%d_%d.png' % (
                                                                    num_bins,
                                                                    k * 100,
                                                                    hs,
                                                                    hr),
                    results)

            print 'RGB'
            im = image.copy()
            desc = mem.cache(descriptors.extract_color_descriptors)(im)
            descs = desc.astype(float)
            ms = mem.cache(cluster.meanshift)(descs, k, hs=hs, hr=hr)
            labels = ms.labels_
            num = np.sqrt(len(labels))
            results = labels.reshape((num, num)).T
            results = ms.cluster_centers_[results][:, :, :-2]

            imsave('images/lena_color_rgb_meanshift_%d_%d_%d.png' % (k * 100,
                                                                     hs,
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

            imsave('images/lena_color_xyz_meanshift_%d_%d_%d.png' % (k * 100,
                                                                     hs,
                                                                     hr),
                   results)

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

            imsave('images/lena_color_hsv_meanshift_%d_%d_%d.png' % (k,
                                                                     hs,
                                                                     hr),
                   results)
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
#            imsave('images/lena_color_gray_meanshift_%d_%d_%d.png' % (k * 100,
#                                                                      hs,
#                                                                      hr),
#                    results)
