"""Noise ablation on the validation split (proposal Sec. 5.2). Results are merged into
results/noise_ablation.csv, so methods can be run one at a time.

    python -m tsr.noise_ablation                              # all six rows, all sigmas (hours)
    python -m tsr.noise_ablation dncnn_finetuned              # only this method
    python -m tsr.noise_ablation bm3d_scaled_1.25 --sigmas 50 # estimated sigma x 1.25, sigma 50 only
    python -m tsr.noise_ablation --plot                       # only redraw the figure

Runs are pinned to 8 logical CPUs by default (--cores N to change, --cores 0 for all).
"""
import argparse
import csv
import functools

import numpy as np
import psutil

from . import config
from .data import load_manifest
from .denoise import DNCNN_FINETUNED, bm3d_denoise, dncnn_denoise
from .evaluate import evaluate, group_ci

SPLIT = "val"
RESULTS = config.ROOT / "results"
CSV = RESULTS / "noise_ablation.csv"
NPZ = RESULTS / "noise_ablation_correct.npz"
COLUMNS = ["method", "sigma", "top1", "ci_lo", "ci_hi", "nll", "top1_aiua", "top1_vid", "psnr", "ssim", "seconds"]
ABLATION = ["none", "bm3d_oracle", "bm3d_estimated", "bm3d_tuned", "dncnn_pretrained", "dncnn_finetuned"]

# BM3D takes ~6 s per full frame and already uses every core. Restoring only the window
# plus a 64 px margin takes ~1.5 s and gave the same predictions and PSNR as the full
# frame on a 48-sample check, because BM3D only searches +-19 px around each block.
BM3D_ROI = 64


def restorer(method, sigma):
    """-> (restore function, roi_margin) for a method name; `sigma` is the true noise level."""
    if method == "none":
        return None, None
    if method == "dncnn_pretrained":
        return dncnn_denoise, None
    if method == "dncnn_finetuned":
        return functools.partial(dncnn_denoise, weights=DNCNN_FINETUNED), None
    if method == "bm3d_oracle":
        return functools.partial(bm3d_denoise, sigma=sigma), BM3D_ROI
    if method == "bm3d_estimated":
        return bm3d_denoise, BM3D_ROI
    if method.startswith("bm3d_scaled_"):  # estimated sigma times a fixed factor
        return functools.partial(bm3d_denoise, scale=float(method.rsplit("_", 1)[1])), BM3D_ROI
    raise ValueError(f"unknown method {method}")


def run(methods, sigmas):
    RESULTS.mkdir(exist_ok=True)
    camera = np.array([r["camera"] for r in load_manifest(SPLIT)])
    for method in methods:
        for sigma in ((0,) if method == "none" else ()) + tuple(sigmas):
            restore, roi = restorer(method, sigma)
            res = evaluate(restore, sigma=sigma, split=SPLIT, roi_margin=roi)
            # Re-read the saved results just before writing, so two runs can overlap.
            table, correct = {}, {}
            if CSV.exists():
                with open(CSV, newline="") as f:
                    table = {(r["method"], int(r["sigma"])): r for r in csv.DictReader(f)}
                correct = dict(np.load(NPZ))
            hit = correct[f"{method}_{sigma}"] = res["correct"]
            lo, hi = group_ci(hit, SPLIT)
            table[(method, sigma)] = dict(
                method=method, sigma=sigma, top1=res["top1"], ci_lo=lo, ci_hi=hi, nll=res["nll"],
                top1_aiua=hit[camera == "aiua"].mean(), top1_vid=hit[camera == "vid"].mean(),
                psnr=res["psnr"], ssim=res["ssim"], seconds=res["seconds"])
            with open(CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, COLUMNS, restval="")
                writer.writeheader()
                writer.writerows(table.values())
            np.savez(NPZ, **correct)
            print(f"{method:17s} sigma {sigma:2d}  top-1 {res['top1']:.4f} [{lo:.3f}, {hi:.3f}]  nll {res['nll']:.4f}  "
                  f"PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}  {res['seconds']:.2f} s/call", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    # name -> (label, colour, dash, marker, x offset so the points do not overlap).
    # Colour is the method family; dashed is the reference variant, solid the deployable one.
    series = {
        "none": ("No restoration", "#52514e", "-", "o", -0.24),
        "bm3d_oracle": ("BM3D, oracle σ", "#2a78d6", (0, (4, 2)), "s", -0.12),
        "bm3d_estimated": ("BM3D, estimated σ", "#2a78d6", "-", "D", 0.0),
        "dncnn_pretrained": ("DnCNN-B, pretrained", "#eb6834", (0, (4, 2)), "^", 0.12),
        "dncnn_finetuned": ("DnCNN-B, fine-tuned", "#eb6834", "-", "v", 0.24),
    }
    slot = {str(s): i for i, s in enumerate(config.NOISE_SIGMAS)}  # evenly spaced noise levels
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    clean = 100 * float(next(r for r in rows if r["sigma"] == "0")["top1"])
    ax.axhline(clean, color=muted, lw=1, ls=(0, (2, 3)))
    ax.text(2.4, clean + 1.5, f"clean {clean:.1f}%", color="#52514e", ha="right", va="bottom", fontsize=9)
    for name, (label, colour, dash, marker, dx) in series.items():
        pts = sorted((r for r in rows if r["method"] == name and r["sigma"] != "0"), key=lambda r: int(r["sigma"]))
        x = np.array([slot[r["sigma"]] for r in pts]) + dx
        y = 100 * np.array([float(r["top1"]) for r in pts])
        err = 100 * np.array([[float(r["top1"]) - float(r["ci_lo"]), float(r["ci_hi"]) - float(r["top1"])] for r in pts]).T
        ax.errorbar(x, y, yerr=err, color=colour, ls=dash, lw=2, marker=marker, ms=7,
                    mec=surface, mew=1.5, elinewidth=1, capsize=0, label=label)
    ax.set_xticks(range(len(slot)), list(slot))
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Noise level σ (0–255 scale)", color="#52514e")
    ax.set_ylabel("Top-1 accuracy (%)", color="#52514e")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=muted, length=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([h[0] for h in handles], labels, frameon=False, loc="lower left", labelcolor=ink, handlelength=3)
    n = len(load_manifest(SPLIT))
    ax.set_title(f"Sign recognition after denoising, LISA validation split ({n} signs)\n",
                 loc="left", color=ink, fontsize=11, fontweight="bold")
    ax.text(0, 1.03, "Frozen ResNet-18; bars are 95% intervals from resampling road segments",
            transform=ax.transAxes, color="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "noise_ablation.png", dpi=200, facecolor=surface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", default=ABLATION)
    parser.add_argument("--sigmas", type=int, nargs="+", default=config.NOISE_SIGMAS)
    parser.add_argument("--plot", action="store_true", help="only redraw the figure")
    parser.add_argument("--cores", type=int, default=8,
                        help="logical CPUs this run may use (0 = all). bm3d ignores OMP_NUM_THREADS "
                             "and otherwise takes every core, which stalls the rest of the machine.")
    args = parser.parse_args()
    if args.cores:
        psutil.Process().cpu_affinity(list(range(args.cores)))
    if not args.plot:
        run(args.methods, args.sigmas)
    plot()
