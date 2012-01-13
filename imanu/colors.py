import numpy as np

from matplotlib.pyplot import imread

from skimage.color.colorconv import rgb2xyz, xyz2rgb


def xyz2luv(arr):
    """
    Converts an RGB image into an LUV

    params
    -------
        image: ndarray (X, Y, 3)
    """
    out = np.empty_like(arr)
    var_U = (4 * image[:, :, 0]) / (image[:, :, 0] + (15 * image[:, :, 1]) + \
            (3 * image[:, :, 2]))
    var_V = (9 * image[:, :, 1]) / (image[:, :, 0] + (15 * image[:, :, 1]) + \
            (3 * image[:, :, 2]))

    var_Y = image[:, :, 1] / 100
    calc =  (var_Y > 0.008856)
    var_Y[calc] = var_Y[calc] ** (1/3)
    var_Y[1 - calc] = 7.787 * var_Y[1 - calc] + 16 / 116

    ref_X =  95.047        # Observer= 2, Illuminant= D65
    ref_Y = 100.000
    ref_Z = 108.883

    ref_U = (4 * ref_X) / (ref_X + (15 * ref_Y) + (3 * ref_Z))
    ref_V = (9 * ref_Y) / (ref_X + (15 * ref_Y) + (3 * ref_Z))

    out[:, :, 0] = (116 * var_Y) - 16
    out[:, :, 1] = 13 * out[:, :, 0] * (var_U - ref_U)
    out[:, :, 2] = 13 * out[:, :, 0] * (var_V - ref_V)
    return out


def rgb2luv(image, un=0.2009, vn=0.4610):
    """
    Converts an RGB image into an LUV

    params
    -------
        image: ndarray (X, Y, 3)
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


def luv2rgb(image, un=0.2009, vn=0.4610):
    """
    Convers an LUV image to RGB

    params:
        image: ndarray (X, Y, 3)
    """
    image_xyz = np.zeros(image.shape)
    Yn = rgb2xyz(np.ones((1, 1, 3)).astype(float))[0][0][2]
    calc = image[:, :, 0] <= 8
    image_xyz[calc, :, 1] = Yn * image[calc, :, 0] * (3. / 29)**3
    calc = image[:, :, 0] > 8
    image_xyz[calc, :, 1] = Yn * ((image[calc, :, 0] + 16.) / 116)**3

    u = image[:, :, 1] / (13 * image[:, :, 0]) + un
    v = image[:, :, 2] / (13 * image[:, :, 0]) + un


    image_xyz[:,:, 0] = image_xyz[:, :, 1] * 9 * u / v
    image_xyz[:,:, 2] = (12 - 3 * u - 20 * v) / (4 * v)
    return xyz2rgb(image_xyz)



if __name__ == "__main__":
    baboon = 1 - imread('./data/baboon.jpg')[::-1].astype(float)
    image = baboon.copy()

