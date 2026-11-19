"""Clean frames, classification windows and the sample manifest."""
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from . import config

_INT_COLUMNS = ("label", "x1", "y1", "x2", "y2", "cx", "cy", "group")
_BILINEAR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16


def load_clean_frame(file):
    """Clean reference frame as uint8 HxWx3. `file` is the path stored in the manifest."""
    frame = np.asarray(Image.open(config.LISA / file).convert("RGB"))
    if config.DEMOSAIC_AIUA and file.startswith("aiua"):
        # Bilinear demosaicing to luminance: the 3x3 kernel has zeros at the three
        # frequencies where a 2x2 mosaic pattern lives, so the checkerboard is removed.
        grey = convolve(frame[..., 0].astype(np.float32), _BILINEAR, mode="mirror")
        frame = np.repeat(np.rint(grey).astype(np.uint8)[..., None], 3, axis=2)
    return frame


def crop_window(frame, cx, cy, size=config.WINDOW):
    """size x size window centred on (cx, cy), reflect-padded at frame borders. No resizing."""
    half = size // 2
    padded = np.pad(frame, ((half, half), (half, half), (0, 0)), mode="reflect")
    return padded[cy:cy + size, cx:cx + size]


def estimate_sigma(frame):
    """Noise std estimate: median(|d|) / 0.6745 over finest-scale diagonal (Haar) coefficients."""
    f = frame.astype(np.float32)
    f = f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2]
    d = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2
    return float(np.median(np.abs(d)) / 0.6745)


def load_manifest(split=None):
    """Rows of data/manifest.csv as dicts; optionally only 'train', 'val' or 'test'."""
    with open(config.MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in _INT_COLUMNS:
            r[c] = int(r[c])
    return [r for r in rows if split is None or r["split"] == split]
