#!/usr/bin/env python3
"""Build a CLASS-AGNOSTIC (single 'object' class) YOLO dataset for Phase-1 localization.

- Collapses every GT box (classes 0..26) to a single class id 0 = 'object'.
- Reuses the EXACT Phase-2 split: val = the even-stride 1000-frame held-out query
  (np.linspace over the sorted frame list), train = every other labeled frame.
  This lets Phase-1 recall compose with the Phase-2 numbers on the same test frames.
- Images are symlinked (not copied); labels are rewritten with class 0.
"""
import os
from pathlib import Path
import numpy as np

DATA = Path("/home/retrocausal-train/Documents/RPN+DinoV2/datasets/gas_valve_2view")
OUT = Path("/home/retrocausal-train/Documents/RPN+DinoV2/phase1/dataset")
N_VAL = 1000


def list_frames(data_dir):
    stems = []
    for p in sorted(data_dir.glob("*.png")):
        if p.with_suffix(".txt").exists():
            stems.append(p.stem)
    return stems


def read_boxes(txt):
    out = []
    for line in Path(txt).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = parts            # drop the original class id
        out.append((xc, yc, w, h))
    return out


def main():
    stems = list_frames(DATA)
    n = len(stems)
    val_idx = sorted(set(np.linspace(0, n - 1, N_VAL).round().astype(int).tolist()))
    val_set = set(val_idx)
    print(f"frames: {n}  ->  val: {len(val_set)}  train: {n - len(val_set)}")

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_boxes = {"train": 0, "val": 0}
    empty = {"train": 0, "val": 0}
    for i, stem in enumerate(stems):
        split = "val" if i in val_set else "train"
        img_link = OUT / "images" / split / f"{stem}.png"
        if not img_link.exists():
            os.symlink(DATA / f"{stem}.png", img_link)
        boxes = read_boxes(DATA / f"{stem}.txt")
        lines = [f"0 {xc} {yc} {w} {h}" for (xc, yc, w, h) in boxes]
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n_boxes[split] += len(boxes)
        if not boxes:
            empty[split] += 1

    yaml = (
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [object]\n"
    )
    (OUT / "data.yaml").write_text(yaml)
    print(f"boxes  train={n_boxes['train']}  val={n_boxes['val']}")
    print(f"empty  train={empty['train']}  val={empty['val']}")
    print(f"wrote {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
