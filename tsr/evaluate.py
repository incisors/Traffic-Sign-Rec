"""Score one restoration method under one degradation with the frozen classifier.

    python -m tsr.evaluate     # no-restoration baseline on the validation split

The test split is kept for one final evaluation (proposal Sec. 5.6); develop on "val".
"""
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

from . import config
from .classifier import load_classifier, logits
from .data import crop_window, load_clean_frame, load_manifest
from .degrade import degrade


def psnr(ref, img):
    mse = np.mean((ref.astype(np.float32) - img) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def group_ci(correct, split, draws=2000):
    """95% interval for top-1, resampling whole road-segment groups: frames of one group
    show the same physical sign, so they are not independent samples.

    Pass the difference of two methods' `correct` arrays (as ints) for a paired interval
    of their accuracy difference on the same samples."""
    groups = np.array([r["group"] for r in load_manifest(split)])
    ids = np.unique(groups)
    hits = np.array([correct[groups == g].sum() for g in ids])
    sizes = np.array([(groups == g).sum() for g in ids])
    pick = np.random.default_rng(0).integers(0, len(ids), (draws, len(ids)))
    return np.percentile(hits[pick].sum(1) / sizes[pick].sum(1), [2.5, 97.5])


def _restore_region(frame, r, margin, restore):
    """Restore only the classification window plus `margin` pixels on each side."""
    half = config.WINDOW // 2 + margin
    y0, y1 = max(0, r["cy"] - half), min(frame.shape[0], r["cy"] + half)
    x0, x1 = max(0, r["cx"] - half), min(frame.shape[1], r["cx"] + half)
    out = frame.copy()
    out[y0:y1, x0:x1] = restore(frame[y0:y1, x0:x1])
    return out


def evaluate(restore=None, sigma=0, blur_len=0, split="test", roi_margin=None):
    """Degrade each full clean frame, restore it, crop the sign windows, classify.

    restore: function(uint8 HxWx3 image) -> uint8 HxWx3 image, or None for no restoration.
    roi_margin: None restores the full frame (the proposal's design). An integer restores
        only the window plus that margin around each sign, which is much faster for slow
        methods; a blind method then estimates its parameters from that region.
    Returns top1, nll (the classifier's cross-entropy, a smoother score than top1), mean
    window psnr / ssim against the clean window, mean seconds per restore call, and
    `correct` (one bool per row of load_manifest(split)).
    """
    rows = load_manifest(split)
    windows, psnrs, ssims, seconds = [], [], [], []
    file = None
    for r in rows:
        if r["file"] != file:
            file = r["file"]
            clean = load_clean_frame(file)
            degraded = frame = degrade(clean, file, sigma, blur_len)
            if restore is not None and roi_margin is None:
                start = time.perf_counter()
                frame = restore(degraded)
                seconds.append(time.perf_counter() - start)
        if restore is not None and roi_margin is not None:
            start = time.perf_counter()
            frame = _restore_region(degraded, r, roi_margin, restore)
            seconds.append(time.perf_counter() - start)
        window, ref = crop_window(frame, r["cx"], r["cy"]), crop_window(clean, r["cx"], r["cy"])
        windows.append(window)
        psnrs.append(psnr(ref, window))
        ssims.append(structural_similarity(ref, window, channel_axis=2, data_range=255))
    scores = logits(load_classifier(), np.stack(windows))
    labels = torch.tensor([r["label"] for r in rows])
    correct = (scores.argmax(1) == labels).numpy()
    return dict(top1=float(correct.mean()), nll=float(torch.nn.functional.cross_entropy(scores, labels)),
                psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)),
                seconds=float(np.mean(seconds)) if seconds else 0.0, correct=correct)


if __name__ == "__main__":
    rows = load_manifest("val")
    cams = {c: np.array([r["camera"] == c for r in rows]) for c in ("aiua", "vid")}
    print(f"val ({len(rows)} samples), no restoration")
    print("  sigma   top-1    aiua     vid    PSNR    SSIM")
    for sigma in (0,) + config.NOISE_SIGMAS:
        res = evaluate(sigma=sigma, split="val")
        print(f"  {sigma:5d}  {res['top1']:.4f}  {res['correct'][cams['aiua']].mean():.4f}  "
              f"{res['correct'][cams['vid']].mean():.4f}  {res['psnr']:6.2f}  {res['ssim']:.4f}")
