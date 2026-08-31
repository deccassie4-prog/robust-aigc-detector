# -*- coding: utf-8 -*-
"""
batch_transform.py — batch image transformation tool for robustness evaluation

Pick a folder (e.g. FOLDER2) and the tool generates 6 sibling folders:
    FOLDER2_JPEG    JPEG compression
    FOLDER2_BLUR    Gaussian blur
    FOLDER2_RESIZE  downscaling
    FOLDER2_NOISE   Gaussian noise
    FOLDER2_COLOR   color jitter
    FOLDER2_CROP    crop
All images under the source folder (including every subdirectory) are transformed
and placed at the same relative path inside each output folder.

The default parameter values below target the competition's transformation subset
(see the README); tune the constants as needed.
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import tkinter as tk
from tkinter import filedialog

# ===================== Tunable parameters =====================
JPEG_QUALITY   = 30      # JPEG compression quality (1-95; lower = heavier artifacts)
BLUR_RADIUS    = 2.0     # Gaussian blur radius (pixels)
SCALE_FACTOR   = 0.5     # downscale factor (0.5 = half size per side)
NOISE_STD      = 20      # Gaussian noise std-dev on the 0-255 range (0-50; higher = noisier)
CROP_KEEP      = 0.8     # kept area fraction for cropping (0.8 = randomly drop 20%)
RANDOM_SEED    = None    # set an integer (e.g. 42) for reproducible runs; None = random each time
# ===================================================

# output folder suffix per transformation
TASKS = {
    "JPEG":    "_JPEG",
    "BLUR":    "_BLUR",
    "RESIZE":  "_RESIZE",
    "NOISE":   "_NOISE",
    "COLOR":   "_COLOR",
    "CROP":    "_CROP",
}

VALID_EXTS = {".png", ".jpg", ".jpeg"}


# ---------------- six transformations ----------------

def tf_jpeg(img):
    """JPEG compression: output is always .jpg; saving at low quality creates compression artifacts"""
    return img, ".jpg"


def tf_blur(img):
    """Gaussian blur"""
    return img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)), None


def tf_resize(img):
    """Downscale by SCALE_FACTOR (Lanczos resampling)"""
    w, h = img.size
    nw, nh = max(1, int(w * SCALE_FACTOR)), max(1, int(h * SCALE_FACTOR))
    return img.resize((nw, nh), Image.LANCZOS), None


def tf_noise(img):
    """Add Gaussian noise"""
    rng = np.random.default_rng()
    arr = np.asarray(img).astype(np.float32)
    noise = rng.normal(0.0, NOISE_STD, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), None


def tf_color(img):
    """Color jitter: random brightness / contrast / saturation perturbation"""
    b = random.uniform(0.6, 1.4)   # brightness
    c = random.uniform(0.6, 1.4)   # contrast
    s = random.uniform(0.6, 1.4)   # saturation
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img, None


def tf_crop(img):
    """Random crop keeping CROP_KEEP of the area"""
    w, h = img.size
    nw = max(1, int(w * CROP_KEEP))
    nh = max(1, int(h * CROP_KEEP))
    x = random.randint(0, w - nw)
    y = random.randint(0, h - nh)
    return img.crop((x, y, x + nw, y + nh)), None


TRANSFORMS = {
    "JPEG":     tf_jpeg,
    "BLUR":     tf_blur,
    "RESIZE":   tf_resize,
    "NOISE":    tf_noise,
    "COLOR":    tf_color,
    "CROP":     tf_crop,
}


# ---------------- main flow ----------------

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # pop up a folder selection dialog
    tk_root = tk.Tk()
    tk_root.withdraw()
    tk_root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select the image folder to process")
    tk_root.destroy()

    if not folder:
        print("No folder selected, exiting.")
        return

    folder = os.path.normpath(folder)
    parent, name = os.path.split(folder)
    print(f"Source folder: {folder}")

    # create the 6 output folders
    targets = {}
    for task, suffix in TASKS.items():
        t = os.path.join(parent, name + suffix)
        os.makedirs(t, exist_ok=True)
        targets[task] = t
        print(f"Output folder: {t}")

    # walk all images in every subdirectory
    count, failed = 0, 0
    for dirpath, _, filenames in os.walk(folder):
        rel_dir = os.path.relpath(dirpath, folder)  # relative path, keeps the directory structure
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in VALID_EXTS:
                continue
            src = os.path.join(dirpath, fn)
            try:
                img = Image.open(src).convert("RGB")
            except Exception as e:
                print(f"[skip] cannot read {src}: {e}")
                failed += 1
                continue

            stem = os.path.splitext(fn)[0]
            for task, tf in TRANSFORMS.items():
                out_img, forced_ext = tf(img)
                out_ext = forced_ext if forced_ext else ext
                out_dir = os.path.join(targets[task], rel_dir)
                os.makedirs(out_dir, exist_ok=True)  # make sure the subdirectory exists
                out_path = os.path.join(out_dir, stem + out_ext)
                if out_ext == ".jpg":
                    out_img.save(out_path, "JPEG", quality=JPEG_QUALITY)
                else:
                    out_img.save(out_path)
            count += 1
            print(f"processed: {os.path.join(rel_dir, fn)}")

    print(f"\nDone! {count} images processed" + (f", {failed} failed." if failed else "."))


if __name__ == "__main__":
    main()
