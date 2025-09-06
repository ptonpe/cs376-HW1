#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """

    return None


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata4.npy")


    pdb.set_trace()
