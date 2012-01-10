from skimage.feature.hog import hog


def get_patch(image, size=9):
    """
    Yields patch of size * size over the whole image

    Parameters
    ----------
    image: ndarray

    size: int, optional
        size of the patches
    """
    h, w = image.shape
    for i in range(w - size):
        for j in range(h - size):
            yield image[j:j + size, i:i + size], (j, i)


def extract_descriptors(image):
    """
    Extract HOGs for patchs of size 9*9 over all the image
    """
    gen = get_patch(image)
    desc = []
    for patch, coord in gen:
        desc.append(hog(patch))
    return desc
