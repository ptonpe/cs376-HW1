#!/usr/bin/python

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless-safe
import matplotlib.pyplot as plt
import cv2
import pdb

def render_2_1(infile="mysterydata/mysterydata2.npy", channels=(0, 4), outprefix="vis2_corrected"):
    X = np.load(infile)
    for ch in channels:
        a = np.log1p(X[:, :, ch].astype(np.float64))
        a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)  # normalize
        plt.imsave(f"{outprefix}_ch{ch}.png", a) 

# --- 2.2: data with NaNs/±Inf → set vmin/vmax explicitly and save ---
def render_2_2(infile="mysterydata/mysterydata3.npy", channels=(1, 7), outprefix="vis3_nanfixed"):
    X = np.load(infile)
    for ch in channels:
        a = X[:, :, ch].astype(np.float64)
        vmin, vmax = np.nanmin(a), np.nanmax(a)
        a = np.nan_to_num(a, nan=vmin, posinf=vmax, neginf=vmin)          # optional but safe
        plt.imsave(f"{outprefix}_ch{ch}.png", a, vmin=vmin, vmax=vmax)


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    X = np.asarray(X)
    colors = np.asarray(colors, dtype=np.float64)
    H, W = X.shape
    N = colors.shape[0]

    vmin = np.nanmin(X)
    vmax = np.nanmax(X)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        idx = np.zeros((H, W), dtype=np.int32)
    else:
        t = (X - vmin) / (vmax - vmin)
        t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
        t = np.clip(t, 0.0, 1.0)
        idx = np.floor(t * (N - 1)).astype(np.int32)

    rgb = colors[idx]
    img = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    # 2.1
    render_2_1("mysterydata/mysterydata2.npy", channels=(0, 4))
    # 2.2
    render_2_2("mysterydata/mysterydata3.npy", channels=(1, 7))
    # 2.3 (uses your colors + mysterydata4)
    colors = np.load("mysterydata/colors.npy")
    X4 = np.load("mysterydata/mysterydata4.npy")
    for ch in range(X4.shape[2]):
        img = colormapArray(X4[:, :, ch], colors)
        plt.imsave(f"vis4_colormap_ch{ch}.png", img)
