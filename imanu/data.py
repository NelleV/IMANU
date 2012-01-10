import numpy as np
from matplotlib.pyplot import imread

def baboon(small=False):
    if small:
        baboon = imread('./data/baboon_small.jpg')[::-1]
    else:
        baboon = imread('./data/baboon.jpg')[::-1]
    return baboon

def python(gray=False):
    python = imread('./data/python.jpg')[::-1]
    return python
