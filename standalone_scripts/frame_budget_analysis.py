#!/usr/bin/env python3
"""How many frames do you need to pick to get at least N samples of every class?

Frames overlap — one frame usually shows several classes at once — so the answer
is (almost always) far fewer than `N * num_classes`. This greedily picks frames
(lazy/accelerated greedy for weighted set cover — exact, not approximate, since
per-frame coverage is a submodular function of the still-unmet quotas) that make
the most progress toward the unmet per-class quotas, one frame at a time, until
every class has its quota or no remaining frame helps.

Works on any flat dataset of images + sibling YOLO `.txt` labels + `classes.txt`
(the same layout `pipeline/dataset.py` expects) — pass `--images-dir`.

This is the analysis/preview tool for the same algorithm `pipeline/dataset.py`'s
`greedy_annotate_split` uses as an actual training-set selection strategy
(`data.sampling: greedy` in a pipeline config) — run this first to pick N, then
flip the config over to get that exact frame set for a real training run.

Usage:
    python standalone_scripts/frame_budget_analysis.py \\
        --images-dir datasets/doosan_swivel --target 30

    # count occurrences (frames) rather than individual boxes per frame:
    python standalone_scripts/frame_budget_analysis.py \\
        --images-dir datasets/doosan_swivel --target 30 --unit boxes

    # save the picked frame stems + full report:
    python standalone_scripts/frame_budget_analysis.py \\
        --images-dir datasets/doosan_swivel --target 30 --out runs/frame_budget.json

    # sweep N=10..50: naive (N*num_classes) vs optimal (greedy) frame counts,
    # one row per N — for plotting:
    python standalone_scripts/frame_budget_analysis.py \\
        --images-dir datasets/doosan_swivel --sweep 10:50 --out runs/frame_budget_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dataset import list_frames, frame_class_counts, greedy_cover  # noqa: E402


def _load_classes(classes_file: Path) -> list[str]:
    return [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]


def sweep(frame_counts: dict[str, Counter], nC: int, lo: int, hi: int) -> list[dict]:
    """{N, naive, optimal} for every N in [lo, hi] — naive assumes zero overlap
    between classes (N * num_classes); optimal re-runs the greedy cover fresh for
    each N (frame_counts is built once and reused, so this stays cheap)."""
    rows = []
    for n in range(lo, hi + 1):
        selected, _ = greedy_cover(frame_counts, nC, [n] * nC)
        rows.append({"N": n, "naive": n * nC, "optimal": len(selected)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default="datasets/doosan_swivel",
                    help="flat dir of images + sibling YOLO .txt labels + classes.txt")
    ap.add_argument("--classes-file", default=None,
                    help="default: <images-dir>/classes.txt")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--target", type=int, default=30,
                    help="minimum samples wanted per class")
    ap.add_argument("--sweep", default=None, metavar="LO:HI",
                    help="instead of a single --target, report naive-vs-optimal frame "
                         "counts for every N in this inclusive range, e.g. '10:50'")
    ap.add_argument("--unit", choices=["frames", "boxes"], default="frames",
                    help="'frames': an image counts once per class it shows "
                         "(default — matches 'N sample images of a class'). "
                         "'boxes': every individual box instance counts.")
    ap.add_argument("--out", default=None,
                    help="write the full report + selected frame stems as JSON here")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    images_dir = images_dir if images_dir.is_absolute() else (REPO_ROOT / images_dir)
    classes_file = Path(args.classes_file) if args.classes_file else images_dir / "classes.txt"
    classes = _load_classes(classes_file)
    nC = len(classes)

    stems = list_frames(images_dir, args.image_ext)
    if not stems:
        raise FileNotFoundError(f"no {args.image_ext} images with sibling .txt labels under {images_dir}")
    frame_counts = frame_class_counts(images_dir, args.image_ext, stems, nC, args.unit)

    if args.sweep:
        lo, sep, hi = args.sweep.partition(":")
        if not sep:
            raise SystemExit(f"--sweep expects 'LO:HI', got {args.sweep!r}")
        lo, hi = int(lo), int(hi)
        rows = sweep(frame_counts, nC, lo, hi)

        print(f"dataset: {images_dir}  ({len(stems)} labeled frames, {nC} classes)\n")
        print(f"{'N':>4s} {'naive':>10s} {'optimal':>10s}")
        for r in rows:
            print(f"{r['N']:>4d} {r['naive']:>10d} {r['optimal']:>10d}")

        if args.out:
            out_path = Path(args.out)
            out_path = out_path if out_path.is_absolute() else (REPO_ROOT / out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "images_dir": str(images_dir), "unit": args.unit,
                "num_classes": nC, "sweep": rows,
            }, indent=2), encoding="utf-8")
            print(f"\n-> {out_path}")
        return 0

    total_available = [0] * nC
    for counts in frame_counts.values():
        for c, n in counts.items():
            total_available[c] += n

    target = [args.target] * nC
    selected, achieved = greedy_cover(frame_counts, nC, target)

    unit_label = "frames showing it" if args.unit == "frames" else "box instances"
    print(f"dataset: {images_dir}  ({len(stems)} labeled frames, {nC} classes)")
    print(f"target: {args.target} {unit_label} per class\n")
    print(f"==> {len(selected)} frames needed to reach the target "
          f"(naive worst case with zero overlap: {sum(target)} frames)\n")

    short = [c for c in range(nC) if total_available[c] < args.target]
    if short:
        print(f"WARNING: {len(short)} class(es) never reach the target — the dataset "
              f"doesn't have enough of them at all:")
        for c in sorted(short, key=lambda c: total_available[c]):
            print(f"  - {classes[c]:<30s} has {total_available[c]}/{args.target}")
        print()

    print(f"{'class':<30s} {'achieved':>9s} {'target':>7s} {'available':>10s}")
    for c in sorted(range(nC), key=lambda c: achieved[c] - target[c]):
        flag = " *" if achieved[c] < target[c] else ""
        print(f"{classes[c]:<30s} {achieved[c]:>9d} {target[c]:>7d} {total_available[c]:>10d}{flag}")

    if args.out:
        out_path = Path(args.out)
        out_path = out_path if out_path.is_absolute() else (REPO_ROOT / out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "images_dir": str(images_dir), "unit": args.unit, "target": args.target,
            "num_frames_needed": len(selected), "selected_frames": selected,
            "per_class": [{"class": classes[c], "achieved": achieved[c],
                          "target": target[c], "available": total_available[c]}
                         for c in range(nC)],
        }, indent=2), encoding="utf-8")
        print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
