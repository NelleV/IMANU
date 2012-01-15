import numpy as np

from skimage.feature.hog import hog
from skimage import color


def get_patch(image, size=9):
    """
    Yields patch of size * size over the whole image

    Parameters
    ----------
    image: ndarray

    size: int, optional
        size of the patches
    """
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape
    for i in range(w - size):
        for j in range(h - size):
            if len(image.shape) == 3:
                yield image[j:j + size, i:i + size, :], (j, i)
            else:
                yield image[j:j + size, i:i + size], (j, i)


def histogram_gray(patch, num_bins=126):
    """
    Create histogram of grays
    """
    return np.histogram(patch, bins=np.arange(0, 256, num_bins))[0]


def extract_descriptors(image, use_hog=True, verbose=True, num_bins=126):
    """
    Extract HOGs for patchs of size 9*9 over all the image
    """
    gen = get_patch(image, size=5)
    descs = []
    for patch, coord in gen:
        if verbose and coord[1] % 5 == 0 and coord[0] == 0:
            print 'computed up to %d, %d' % coord
        if len(patch.shape) == 3:
            if use_hog:
                desc = hog(patch.mean(axis=2))
                np.concatenate((desc, patch.mean(axis=0).mean(axis=0)))
            else:
                desc = np.concatenate((histogram_gray(patch.mean(axis=2),
                                                      num_bins=num_bins),
                                       image[coord]))
        else:
            if use_hog:
                desc = hog(patch)
            else:
                desc = np.concatenate((histogram_gray(patch,
                                                      num_bins=num_bins),
                                       image[coord]))
        desc = np.concatenate((desc, np.array(coord)))
        descs.append(desc)
    return np.array(descs)


def extract_color_descriptors(image, space=None, verbose=False):
    """
    Extract color descriptors of the image

    Parameters
    ----------
    image:

    space:
        None, equivalent to rgb
        rgb
        hsv
        xyz
        rgbcie

    Returns
    -------
    Descriptors
    """
    if space == 'hsv':
        image = color.rgb2hsv(image)
    elif space == 'xyz':
        image = color.rgb2xyz(image)
    elif space == 'rgbcie':
        image = color.rgb2rgbcie(image)
    elif space == 'gray' or space == 'grey':
        image = color.rgb2gray(image)

    gen = get_patch(image, size=1)
    descs = []
    for patch, coord in gen:
        if verbose and coord[1] % 5 == 0 and coord[0] == 0:
            print 'computed up to %d, %d' % coord
        desc = patch.flatten()
        desc = np.concatenate((desc, np.array(coord)))
        descs.append(desc)
    return np.array(descs)
