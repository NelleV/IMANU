from skimage import data
from matplotlib.pyplot import imread
from scipy.misc import imresize


def baboon(small=False):
    if small:
        baboon = imread('./data/baboon_small.jpg')[::-1]
    else:
        baboon = imread('./imanu/data/baboon.jpg')[::-1]
    return baboon


def chinese():
    return imread('./imanu/data/chinese_temple.png')


def white_baboon():
    return imread('./imanu/data/white_baboon.png')


def black_swan():
    return imread('./imanu/data/black_swan.png')


def python(gray=False):
    python = imread('./data/python.jpg')[::-1]
    return python


def lena(gray=True, small=True):
    im = data.lena()
    if small:
        im = imresize(im, size=(im.shape[0] / 4, im.shape[1] / 4))
    if gray:
        im = im.mean(axis=2)
    return im
