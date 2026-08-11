"""Frame listing, train/val split, annotation-budget subsetting, and building the
class-agnostic single-class YOLO dataset.

Expects a flat directory of images (`*.png`) each with a sibling YOLO label file
(`*.txt`, `class xc yc w h` normalized) plus a `classes.txt`.
"""
from __future__ import annotations

import heapq
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def list_frames(images_dir: Path, ext: str = "png") -> list[str]:
    """Sorted stems that have both an image and a label file."""
    return [p.stem for p in sorted(images_dir.glob(f"*.{ext}"))
            if p.with_suffix(".txt").exists()]


def _even_idx(n: int, k: int) -> list[int]:
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def annotate_split(stems: list[str], n_annotate: int, sampling: str = "even",
                   seed: int = 0) -> tuple[list[str], list[str]]:
    """Pick `n_annotate` frames (the manually-annotated set, spread evenly across the
    dataset) and return (annotated, remaining). `remaining` is what the pipeline
    auto-annotates and scores."""
    n = len(stems)
    if not 0 < n_annotate < n:
        raise ValueError(f"annotation_frames must be in (0, {n}); got {n_annotate}")
    if sampling == "even":
        idx = _even_idx(n, n_annotate)
    else:
        idx = sorted(np.random.default_rng(seed).choice(n, n_annotate, replace=False).tolist())
    aset = set(idx)
    annotated = [stems[i] for i in idx]
    remaining = [stems[i] for i in range(n) if i not in aset]
    return annotated, remaining


def frame_class_counts(images_dir: Path, ext: str, stems: list[str], nC: int,
                       unit: str = "frames") -> dict[str, Counter]:
    """stem -> Counter(class_id -> count), for every frame with at least one box.
    `unit='frames'` caps each class at 1 per frame (an image either shows the
    class or it doesn't); `unit='boxes'` counts every individual instance."""
    out = {}
    for s in stems:
        counts = Counter()
        for row in _read_label(images_dir / f"{s}.txt"):
            c = int(float(row[0]))
            if 0 <= c < nC:
                counts[c] += 1
        if unit == "frames":
            counts = Counter({c: 1 for c in counts})
        if counts:
            out[s] = counts
    return out


def greedy_cover(frame_counts: dict[str, Counter], nC: int,
                 target: list[int]) -> tuple[list[str], list[int]]:
    """Lazy greedy weighted set cover — exact, not approximate, since per-frame
    coverage is a submodular function of the still-unmet per-class quotas.
    Repeatedly picks whichever frame currently closes the most total unmet quota,
    until every class hits its target or no remaining frame helps. Returns
    (selected_stems_in_pick_order, achieved_per_class)."""
    remaining = list(target)

    def value(counts: Counter) -> int:
        return sum(min(remaining[c], n) for c, n in counts.items() if remaining[c] > 0)

    heap = [(-value(counts), stem) for stem, counts in frame_counts.items()]
    heap = [h for h in heap if h[0] < 0]
    heapq.heapify(heap)

    achieved = [0] * nC
    selected = []
    while heap and any(remaining):
        neg_val, stem = heapq.heappop(heap)
        counts = frame_counts[stem]
        cur_val = value(counts)
        if cur_val <= 0:
            continue
        if heap and cur_val < -heap[0][0]:
            heapq.heappush(heap, (-cur_val, stem))  # stale — re-rank and retry
            continue
        selected.append(stem)
        for c, n in counts.items():
            take = min(remaining[c], n)
            remaining[c] -= take
            achieved[c] += take
    return selected, achieved


def greedy_annotate_split(images_dir: Path, ext: str, stems: list[str], nC: int,
                          target_per_class: int) -> tuple[list[str], list[str]]:
    """Pick the annotated (training) set via greedy weighted set cover instead of
    even/random spacing: repeatedly pick whichever frame currently closes the most
    unmet per-class quota, until every class has `target_per_class` sample frames
    (or the dataset runs out for that class). Frames overlap — most show several
    classes at once — so this reaches the quota with far fewer frames than
    even/random sampling would need to hit the same per-class coverage by chance.
    Same algorithm as `standalone_scripts/frame_budget_analysis.py`, exposed here
    so the pipeline can use it as an actual selection strategy, not just analysis."""
    frame_counts = frame_class_counts(images_dir, ext, stems, nC, unit="frames")
    selected, _ = greedy_cover(frame_counts, nC, [target_per_class] * nC)
    aset = set(selected)
    remaining = [s for s in stems if s not in aset]
    return selected, remaining


def even_subset(stems: list[str], k: int, sampling: str = "even",
                seed: int = 0) -> list[str]:
    """Evenly-spread subset of `k` frames. k<=0 or k>=len -> all frames."""
    if k is None or k <= 0 or k >= len(stems):
        return list(stems)
    if sampling == "even":
        return [stems[i] for i in _even_idx(len(stems), k)]
    return [stems[i] for i in sorted(np.random.default_rng(seed).choice(len(stems), k, replace=False).tolist())]


def _read_label(txt: Path) -> list[list[str]]:
    """Full YOLO rows [cls, xc, yc, w, h] as strings."""
    return [p for p in (ln.split() for ln in txt.read_text().splitlines()) if len(p) == 5]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink `dst` -> `src`; fall back to a hardlink, then a plain copy,
    when symlinks aren't permitted (e.g. Windows without Developer Mode)."""
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _shuffled_names(stems: list[str], seed: int) -> list[str]:
    """New zero-padded sequential names for `stems`, shuffled to a random
    position — same content, different sort-order position. Used to test whether
    on-disk listing order (which anything doing `sorted(glob(...))`, e.g. a
    trainer's initial directory scan, depends on) affects training, independent
    of which frames are actually used."""
    width = max(len(s) for s in stems)
    order = list(range(len(stems)))
    random.Random(seed).shuffle(order)
    new_names = [None] * len(stems)
    for new_idx, orig_idx in enumerate(order):
        new_names[orig_idx] = str(new_idx + 1).zfill(width)
    return new_names


def build_yolo_dataset(images_dir: Path, ext: str, train_stems: list[str],
                       val_stems: list[str], out_dir: Path,
                       names: list[str], shuffle_train_names: bool = False,
                       seed: int = 0) -> Path:
    """Symlink images and write labels, keeping original class ids (nc=len(names)).

    `shuffle_train_names`: write the train split under randomized filenames
    (deterministic given `seed`) instead of the original stems — the exact same
    frames/labels, just a different on-disk name/listing order. Isolates whether
    frame *order* (as opposed to frame *selection*) affects training. Val split
    always keeps its original stems (it's the detector's per-epoch monitor set,
    not part of this question)."""
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_names = _shuffled_names(train_stems, seed) if shuffle_train_names else train_stems

    for split, stems, out_names in (("train", train_stems, train_names),
                                    ("val", val_stems, val_stems)):
        for s, out_name in zip(stems, out_names):
            link = out_dir / "images" / split / f"{out_name}.{ext}"
            if link.exists() or link.is_symlink():
                link.unlink()
            _link_or_copy((images_dir / f"{s}.{ext}").resolve(), link)
            lines = [f"{p[0]} {' '.join(p[1:])}"
                     for p in _read_label(images_dir / f"{s}.txt")]
            (out_dir / "labels" / split / f"{out_name}.txt").write_text("\n".join(lines))

    nc, names_yaml = len(names), "[" + ", ".join(names) + "]"
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {nc}\nnames: {names_yaml}\n"
    )
    return data_yaml
