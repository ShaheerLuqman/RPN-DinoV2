#!/usr/bin/env python3
"""Build a copy of a flat image+YOLO-label dataset with every frame renamed to a
random position — for testing whether frame *order* affects `sampling: even`.

`even` sampling (`pipeline/dataset.py:_even_idx`) picks indices evenly spread
across the *sorted filename* order. When filenames are sequential frame numbers
from a video, sorted order == temporal order, so "evenly spread" also means
"evenly spread across the video timeline" — real coverage diversity. This script
breaks that correspondence: every frame keeps its original labels, but gets a new
random filename, so picking indices evenly spread across the new sorted order no
longer corresponds to anything about the original video. Point a config's
`data.images_dir` at the output and run `sampling: even` again — if results
change materially, frame order (not just frame count) was doing real work.

Images/labels are hardlinked (falling back to symlink, then copy) — cheap, no
duplicated pixel data, matches `pipeline.dataset._link_or_copy`'s fallback order
reversed for Windows-without-symlink-privilege (hardlink tried first here since
that's what actually works without admin rights).

Usage:
    python standalone_scripts/shuffle_dataset.py \\
        --images-dir datasets/doosan_swivel --out-dir datasets/doosan_swivel_shuffled
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dataset import list_frames  # noqa: E402


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """Hardlink `dst` -> `src`; fall back to a symlink, then a plain copy."""
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                    help="flat dir of images + sibling YOLO .txt labels + classes.txt")
    ap.add_argument("--out-dir", required=True, help="where the renamed copy is built")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    images_dir = images_dir if images_dir.is_absolute() else (REPO_ROOT / images_dir)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else (REPO_ROOT / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = list_frames(images_dir, args.image_ext)
    if not stems:
        raise FileNotFoundError(f"no {args.image_ext} images with sibling .txt labels under {images_dir}")

    width = max(len(s) for s in stems)  # keep the source's zero-padded width
    order = stems.copy()
    random.Random(args.seed).shuffle(order)

    mapping = {}
    for i, old_stem in enumerate(order):
        new_stem = str(i + 1).zfill(width)
        _hardlink_or_copy((images_dir / f"{old_stem}.{args.image_ext}").resolve(),
                          out_dir / f"{new_stem}.{args.image_ext}")
        _hardlink_or_copy((images_dir / f"{old_stem}.txt").resolve(),
                          out_dir / f"{new_stem}.txt")
        mapping[new_stem] = old_stem

    classes_src = images_dir / "classes.txt"
    if classes_src.exists():
        _hardlink_or_copy(classes_src.resolve(), out_dir / "classes.txt")

    (out_dir / "shuffle_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"[shuffle] {len(stems)} frames -> {out_dir} (seed={args.seed})")
    print(f"[shuffle] mapping (new_stem -> original_stem) -> {out_dir / 'shuffle_mapping.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
