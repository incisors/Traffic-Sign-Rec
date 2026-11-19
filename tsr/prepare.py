"""Build data/manifest.csv: the one list of samples and splits everything else reads.

    python -m tsr.prepare
"""
import collections
import csv
import random

import numpy as np
from scipy.ndimage import laplace

from . import config
from .data import load_clean_frame


def read_annotations():
    with open(config.LISA / "allAnnotations.csv", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))[1:]
    recs = []
    for r in rows:
        x1, y1, x2, y2 = map(int, r[2:6])
        recs.append(dict(file=r[0], cls=r[1], x1=x1, y1=y1, x2=x2, y2=y2,
                         cx=(x1 + x2) // 2, cy=(y1 + y2) // 2,
                         camera="aiua" if r[0].startswith("aiua") else "vid",
                         origin=r[7], oframe=int(r[8]), track=r[9]))
    return recs


def assign_groups(recs):
    """Tracks of one video whose frame ranges come within GROUP_GAP frames show the same
    stretch of road, so they must land in the same split."""
    spans = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        spans[r["origin"]][r["track"]].append(r["oframe"])
    group_of, gid = {}, 0
    for origin in sorted(spans):
        end = None
        for lo, hi, track in sorted((min(v), max(v), t) for t, v in spans[origin].items()):
            if end is None or lo > end + config.GROUP_GAP:
                gid += 1
                end = hi
            else:
                end = max(end, hi)
            group_of[(origin, track)] = gid
    for r in recs:
        r["group"] = group_of[(r["origin"], r["track"])]


def split_by_group(recs, classes):
    """Random group-level split; among 300 shuffles keep the one whose val/test sizes are
    closest to target while every class has at least 5 val and 5 test samples."""
    groups = sorted({r["group"] for r in recs})
    n_train = int(config.SPLIT[0] * len(groups))
    n_val = int((config.SPLIT[0] + config.SPLIT[1]) * len(groups))
    best = None
    for seed in range(300):
        g = groups[:]
        random.Random(seed).shuffle(g)
        part = {x: "train" for x in g[:n_train]}
        part.update({x: "val" for x in g[n_train:n_val]})
        part.update({x: "test" for x in g[n_val:]})
        cells = collections.Counter((part[r["group"]], r["cls"]) for r in recs)
        sizes = collections.Counter(part[r["group"]] for r in recs)
        covered = min(cells[(s, c)] for s in ("val", "test") for c in classes) >= 5
        deviation = sum(abs(sizes[s] / len(recs) - t) for s, t in zip(("val", "test"), config.SPLIT[1:]))
        if best is None or (covered, -deviation) > best[0]:
            best = ((covered, -deviation), part)
    for r in recs:
        r["split"] = best[1][r["group"]]


def main():
    recs = read_annotations()
    assign_groups(recs)
    print(f"annotations: {len(recs)}")

    recs = [r for r in recs if r["x2"] - r["x1"] >= config.MIN_BOX and r["y2"] - r["y1"] >= config.MIN_BOX]
    print(f"bbox >= {config.MIN_BOX}: {len(recs)}")

    frames = {}
    for r in recs:
        if r["file"] not in frames:
            frames[r["file"]] = load_clean_frame(r["file"]).mean(axis=2)
        r["sharpness"] = float(laplace(frames[r["file"]][r["y1"]:r["y2"], r["x1"]:r["x2"]]).var())
    del frames
    cut = np.quantile([r["sharpness"] for r in recs], config.SHARPNESS_DROP)
    dropped = collections.Counter(r["camera"] for r in recs if r["sharpness"] < cut)
    total = collections.Counter(r["camera"] for r in recs)
    print(f"sharpness cut {cut:.1f}: dropped " + ", ".join(f"{c} {dropped[c]}/{total[c]}" for c in sorted(total)))
    recs = [r for r in recs if r["sharpness"] >= cut]

    classes = [c for c, _ in collections.Counter(r["cls"] for r in recs).most_common(config.NUM_CLASSES)]
    recs = [r for r in recs if r["cls"] in classes]
    for r in recs:
        r["label"] = classes.index(r["cls"])
    split_by_group(recs, classes)

    columns = ["file", "cls", "label", "x1", "y1", "x2", "y2", "cx", "cy", "camera", "group", "split", "sharpness"]
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    print(f"kept {len(recs)} samples on {len({r['file'] for r in recs})} frames, "
          f"{len({r['group'] for r in recs})} groups -> {config.MANIFEST}")
    cells = collections.Counter((r["split"], r["cls"]) for r in recs)
    cams = collections.Counter((r["split"], r["camera"]) for r in recs)
    print(f"{'class':20s} train   val  test")
    for c in classes:
        print(f"{c:20s} {cells[('train', c)]:5d} {cells[('val', c)]:5d} {cells[('test', c)]:5d}")
    for c in ("aiua", "vid", None):
        row = [sum(v for (s, cam), v in cams.items() if s == sp and c in (cam, None)) for sp in ("train", "val", "test")]
        print(f"{'TOTAL' if c is None else '  camera ' + c:20s} {row[0]:5d} {row[1]:5d} {row[2]:5d}")


if __name__ == "__main__":
    main()
