import numpy as np

from matplotlib.pyplot import imread

from skimage.color.colorconv import rgb2xyz


def rgb2luv(image, un=0.2009, vn=0.4610):
    """
    Converts an RGB image into an LUV

    params
    -------
        image: ndarray
    """
    Yn = rgb2xyz(np.ones((1, 1, 3)).astype(float))[0][0][2]
    image_xyz = rgb2xyz(image)
    image_luv = np.zeros(image.shape)
    u = 4 * image_xyz[:, :, 0] / (image_xyz[:, :, 0] + 15 * image_xyz[:, :, 1] + 3 * \
                                image_xyz[:, :, 2])
    v = 9 * image_xyz[:, :, 1] / (image_xyz[:, :, 1] + 15 * image_xyz[:, :, 1] + 3 * \
                                image_xyz[:, :, 2])

    calc = image[:, :, 1] / Yn <= (6./29)**3
    image_luv[calc, :, 0] = (29./3)**3 * image_xyz[calc, :, 1] / Yn
    calc = image[:, :, 1] / Yn > (6./29)**3
    image_luv[calc, :, 0] = 116 * image_xyz[calc, :, 1]**(-1./3) / Yn

    image_luv[:, :, 1] = 13 * image_luv[:, :, 0] * (u - un)
    image_luv[:, :, 2] = 13 * image_luv[:, :, 0] * (v - vn)
    return image_luv



if __name__ == "__main__":
    baboon = imread('./data/baboon.jpg').astype(float)
    image = baboon.copy()
    baboon = rgb2luv(baboon.astype(float))

