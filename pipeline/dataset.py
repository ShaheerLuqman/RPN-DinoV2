"""Frame listing, train/val split, annotation-budget subsetting, and building the
class-agnostic single-class YOLO dataset.

Expects a flat directory of images (`*.png`) each with a sibling YOLO label file
(`*.txt`, `class xc yc w h` normalized) plus a `classes.txt`.
"""
from __future__ import annotations

import os
import shutil
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


def build_yolo_dataset(images_dir: Path, ext: str, train_stems: list[str],
                       val_stems: list[str], out_dir: Path,
                       names: list[str]) -> Path:
    """Symlink images and write labels, keeping original class ids (nc=len(names))."""
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, stems in (("train", train_stems), ("val", val_stems)):
        for s in stems:
            link = out_dir / "images" / split / f"{s}.{ext}"
            if link.exists() or link.is_symlink():
                link.unlink()
            _link_or_copy((images_dir / f"{s}.{ext}").resolve(), link)
            lines = [f"{p[0]} {' '.join(p[1:])}"
                     for p in _read_label(images_dir / f"{s}.txt")]
            (out_dir / "labels" / split / f"{s}.txt").write_text("\n".join(lines))

    nc, names_yaml = len(names), "[" + ", ".join(names) + "]"
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {nc}\nnames: {names_yaml}\n"
    )
    return data_yaml
