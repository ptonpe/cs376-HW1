"""
colorspaces_tools.py
Helpers for HW Colorspaces section:
- Coding 4.1: RGB channel plots
- Coding 4.2: LAB channel plots
- DIY/Coding 4.3: prepare two images (resize/crop to 256x256) and pick patch coordinates, write info.txt

Usage examples:
  python colorspaces_tools.py <root> --do rgb lab
  python colorspaces_tools.py <root> --im1 my_lamp.jpg --im2 my_window.jpg --prep_4_3 --crop "x,y,w,h" --desc "lamp vs window"
  python colorspaces_tools.py <root> --pick <root>/im1.jpg <root>/im2.jpg --desc "lamp vs window"

Note: requires opencv-python and matplotlib.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def _save_triplet(imgs, titles, out_path, cmap='gray'):
    """Save a 1x3 panel figure."""
    plt.figure(figsize=(9, 3))
    for i, (im, ttl) in enumerate(zip(imgs, titles), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(im, cmap=cmap, vmin=None, vmax=None)
        plt.axis('off')
        plt.title(ttl)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- 4.1: RGB channel plots ----------

def plot_rgb_channels(image_bgr: np.ndarray, name_prefix: str, out_dir: str = "results_colorspaces"):
    """Plot R,G,B channels as grayscale images (3-up panel)."""
    _ensure_dir(out_dir)
    b, g, r = cv2.split(image_bgr)  # OpenCV loads BGR
    r_img, g_img, b_img = r, g, b
    out_path = os.path.join(out_dir, f"{name_prefix}_RGB_channels.png")
    _save_triplet([r_img, g_img, b_img], ["R", "G", "B"], out_path, cmap='gray')
    return out_path

# ---------- 4.2: LAB channel plots ----------

def plot_lab_channels(image_bgr: np.ndarray, name_prefix: str, out_dir: str = "results_colorspaces"):
    """Convert to CIELAB and plot L, a, b channels (3-up panel)."""
    _ensure_dir(out_dir)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)  # OpenCV uses L in [0,255]
    L, a, b = cv2.split(lab)
    out_path = os.path.join(out_dir, f"{name_prefix}_LAB_channels.png")
    _save_triplet([L, a, b], ["L", "a", "b"], out_path, cmap='gray')
    return out_path

# ---------- 4.3: DIY ----------

def resize_crop_to_256(img_bgr: np.ndarray, crop: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Return 256x256 BGR image. If crop=(x,y,w,h) given, crop then resize.
    If not, do a centered crop to square then resize.
    """
    H, W = img_bgr.shape[:2]
    if crop is not None:
        x, y, w, h = crop
        x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        img_bgr = img_bgr[y:y + h, x:x + w]
    else:
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        img_bgr = img_bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

def save_prepared_4_3(im_path: str, out_name: str, crop_box: Optional[Tuple[int, int, int, int]] = None, out_dir: str = ".") -> str:
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {im_path}")
    out = resize_crop_to_256(img, crop=crop_box)
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, out)
    return out_path

def pick_coordinate(image_path: str) -> Tuple[int, int]:
    """Open an image and let user click once; return (x,y) in image coords."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click one point (press Enter to confirm)")
    pts = plt.ginput(1, timeout=0)  # wait until a click
    plt.close(fig)
    if not pts:
        raise RuntimeError("No point clicked.")
    x, y = int(round(pts[0][0])), int(round(pts[0][1]))
    return x, y

def write_info_txt(p1: Tuple[int, int], p2: Tuple[int, int], desc: str, out_path: str = "info.txt"):
    with open(out_path, "w") as f:
        f.write(f"{p1[0]} {p1[1]} {p2[0]} {p2[1]}\n")
        f.write(desc.strip() + "\n")
    return out_path

# ---------- main CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="root containing rubik/ or your images")
    ap.add_argument("--indoor", default="indoor.png")
    ap.add_argument("--outdoor", default="outdoor.png")
    ap.add_argument("--im1", default=None, help="path to your first photo")
    ap.add_argument("--im2", default=None, help="path to your second photo")
    ap.add_argument("--prep_4_3", action="store_true", help="prepare im1/im2 as 256x256 im1.jpg/im2.jpg")
    ap.add_argument("--crop", default=None, help='crop bbox "x,y,w,h" (optional) for both images')
    ap.add_argument("--desc", default="")
    ap.add_argument("--pick", nargs="*", help="paths to images to pick a single (x,y) from each; writes/prints coords")
    ap.add_argument("--do", nargs="*", default=[], help="subset of tasks to run: rgb lab")
    ap.add_argument("--outdir", default="results_colorspaces")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 4.1 / 4.2 on rubik images if requested
    if any(s in args.do for s in ("rgb", "lab")):
        rubik_dir = os.path.join(args.root, "rubik")
        for tag, name in (("indoor", args.indoor), ("outdoor", args.outdoor)):
            path = os.path.join(rubik_dir, name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] cannot read {path}; skipping {tag}")
                continue
            if "rgb" in args.do:
                plot_rgb_channels(img, f"rubik_{tag}", args.outdir)
            if "lab" in args.do:
                plot_lab_channels(img, f"rubik_{tag}", args.outdir)

    # 4.3 prep
    if args.prep_4_3:
        if not args.im1 or not args.im2:
            raise SystemExit("--prep_4_3 requires --im1 and --im2")
        crop_box = None
        if args.crop:
            x, y, w, h = map(int, args.crop.split(","))
            crop_box = (x, y, w, h)
        p1 = save_prepared_4_3(os.path.join(args.root, args.im1), "im1.jpg", crop_box, out_dir=args.root)
        p2 = save_prepared_4_3(os.path.join(args.root, args.im2), "im2.jpg", crop_box, out_dir=args.root)
        print("wrote:", p1, p2)

    # coordinate picking
    if args.pick:
        if len(args.pick) != 2:
            raise SystemExit("--pick expects two image paths (im1 and im2)")
        p1 = pick_coordinate(args.pick[0])
        p2 = pick_coordinate(args.pick[1])
        print("picked:", p1, p2)
        if args.desc:
            write_info_txt(p1, p2, args.desc, out_path=os.path.join(args.root, "info.txt"))
            print("wrote info.txt")
        else:
            print("tip: add --desc 'your lighting description' to auto-write info.txt")

if __name__ == "__main__":
    main()
