import argparse
import os
import sys
import cv2
import numpy as np
import pdb


def coverPalette(N):
    # Return the palette we're using
    return np.linspace(1, 0, 2 ** N)


def linearToSRGB(M):
    """Given a matrix of linear intensities, convert to sRGB
    Adapted from: https://www.nayuki.io/page/gamma-aware-image-dithering"""
    M = np.clip(M, 0, 1)
    mask = (M < 0.0031308).astype(float)
    out1 = 12.92 * M
    out2 = (M ** (1 / 2.4)) * 1.055 - 0.055
    return mask * out1 + (1 - mask) * out2


def SRGBToLinear(M):
    """Given a matrix of sRGB intensities, convert to linear
    Adapted from: https://www.nayuki.io/page/gamma-aware-image-dithering"""
    M = np.clip(M, 0, 1)
    mask = (M < 0.04045).astype(float)
    return mask * (M / 12.92) + (1 - mask) * (((M + 0.055) / 1.055) ** 2.4)


def reconstructImage(IQ, palette):
    """
    Given a quantized image IQ and their value, return a floating point image
    """
    assert np.issubdtype(IQ.dtype, np.integer)
    return palette[IQ]


def upscaleNN(I, target_size):
    """
    NN upsample I until it hits a target size but without going over 4096
    """
    h, w = I.shape[:2]
    scale = 1
    while True:
        if min(h * scale, w * scale) >= target_size:
            break
        if max(h * (scale + 1), w * (scale + 1)) > 4096:
            break
        scale += 1
    shape = (scale, scale) if I.ndim == 2 else (scale, scale, 1)
    return np.kron(I, np.ones(shape))


def resizeToSquare(I, maxDim):
    """Given an image, make sure it's no bigger than maxDim on either side"""
    h, w = I.shape[:2]
    m = max(h, w)
    if m <= maxDim:
        return I
    s = maxDim / float(m)
    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_AREA)


def quantize(v, palette):
    """
    Given a scalar v and array of values palette,
    return the index of the closest value
    """
    pal = np.asarray(palette, dtype=np.float64)                  # (K,)
    a = np.asarray(v, dtype=np.float64)                          # () or (...,)
    diffs = np.abs(a[..., None] - pal[None, :])                  # (..., K)
    idx = np.argmin(diffs, axis=-1)                              # () or (...,)
    return idx.astype(np.uint8)


def quantizeNaive(IF, palette):
    """Given a floating-point image return quantized version (Naive)"""
    I = np.asarray(IF, dtype=np.float64)
    if I.ndim == 2:
        return np.argmin(np.abs(I[..., None] - palette[None, None, :]), axis=2).astype(np.uint8)
    else:
        return np.argmin(np.abs(I[..., None] - palette[None, None, None, :]), axis=3).astype(np.uint8)


def quantizeFloyd(IF, palette):
    """
    Given a floating-point image return quantized version (Floyd-Steinberg)
    """
    P = np.asarray(palette, dtype=np.float64)
    pix = np.asarray(IF, dtype=np.float64).copy()
    H, W = pix.shape[:2]
    C = 1 if pix.ndim == 2 else pix.shape[2]
    out = np.zeros((H, W) if C == 1 else (H, W, C), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            old = pix[y, x].copy()
            idx = quantize(old, P)
            out[y, x] = idx
            new = P[idx]
            err = old - new

            if x + 1 < W:               pix[y,   x+1] += err * (7/16)
            if y + 1 < H and x > 0:     pix[y+1, x-1] += err * (3/16)
            if y + 1 < H:               pix[y+1, x  ] += err * (5/16)
            if y + 1 < H and x + 1 < W: pix[y+1, x+1] += err * (1/16)
    return out


def quantizeFloydGamma(IF, palette, gamma=2.2):
    """
    Given a floating-point image return quantized version
    (Floyd-Steinberg with Gamma Correction)
    """
    P = np.asarray(palette, dtype=np.float64)
    P_lin = np.power(P, gamma)                                   # linearize palette

    pix = np.asarray(IF, dtype=np.float64).copy()
    pix = np.power(np.clip(pix, 0.0, 1.0), gamma)                # linearize image

    H, W = pix.shape[:2]
    C = 1 if pix.ndim == 2 else pix.shape[2]
    out = np.zeros((H, W) if C == 1 else (H, W, C), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            old = pix[y, x].copy()                               # scalar or length-C
            idx = quantize(old, P_lin)                           # per-channel argmin to P_lin
            out[y, x] = idx
            new = P_lin[idx]                                     # scalar or length-C
            err = old - new

            if x + 1 < W:               pix[y,   x+1] += err * (7/16)
            if y + 1 < H and x > 0:     pix[y+1, x-1] += err * (3/16)
            if y + 1 < H:               pix[y+1, x  ] += err * (5/16)
            if y + 1 < H and x + 1 < W: pix[y+1, x+1] += err * (1/16)

    return out


def parse():
    parser = argparse.ArgumentParser(description='run dither')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('algorithm', help="What function to call")
    parser.add_argument('--numbits', default=1, type=int,
                        help="Number of bits to use; play with this!")
    parser.add_argument('--resizeto', default=500, type=int,
                        help="Resize largest side to this (keeps aspect)")
    parser.add_argument('--grayscale', default=1, type=int,
                        help="Whether to grayscale first (1=yes, 0=no)")
    parser.add_argument('--scaleup', default=1000, type=int,
                        help="Downsampling behaves nicer than upsampling")
    parser.add_argument('--color', action='store_true',
                        help="Process in color (H×W×3); overrides --grayscale")
    args = parser.parse_args()
    if args.color:
        args.grayscale = 0
    return args


if __name__ == "__main__":
    args = parse()

    if args.algorithm not in globals():
        print("I don't recognize that algorithm")
        sys.exit(1)

    if not os.path.exists(args.target):
        os.mkdir(args.target)

    images = [fn for fn in os.listdir(args.source) if fn.endswith(".jpg")]
    images.sort()

    algo_fn = globals()[args.algorithm]
    palette = coverPalette(args.numbits)

    heights = {}

    for imageI, image in enumerate(images):
        print("%d/%d" % (imageI, len(images)))

        # Read (always color), then optionally grayscale
        I = cv2.imread(os.path.join(args.source, image), cv2.IMREAD_COLOR)  # BGR uint8
        if I is None:
            print(f"skip unreadable: {image}")
            continue
        I = resizeToSquare(I, args.resizeto)

        I = I.astype(float) / 255.0
        if args.grayscale:
            I = np.mean(I, axis=2)  # H×W

        IQ = algo_fn(I, palette)
        R = reconstructImage(IQ, palette)

        heights[image] = I.shape[0]
        if args.scaleup > 0:
            I, R = upscaleNN(I, args.scaleup), upscaleNN(R, args.scaleup)

        I_path = os.path.join(args.target, image + "_orig.png")
        cv2.imwrite(I_path, (I * 255).astype(np.uint8))
        R_path = os.path.join(args.target, f"{image}_{args.algorithm}.png")
        cv2.imwrite(R_path, (R * 255).astype(np.uint8))

    view_algos = ["orig", args.algorithm]

    with open(os.path.join(args.target, "view.html"), "w") as fh:
        fh.write("<html><body><table>")
        fh.write("<tr>")
        for algo in view_algos:
            fh.write(f"<td>{algo}</td>")
        fh.write("</tr>")
        for image in images:
            height = heights.get(image, 256)
            fh.write("<tr>")
            for algo in view_algos:
                img_path = f"{image}_{algo}.png"
                fh.write(f"<td><img height='{height}' src='{img_path}'></td>")
            fh.write("</tr>")
        fh.write("</table></body></html>")
