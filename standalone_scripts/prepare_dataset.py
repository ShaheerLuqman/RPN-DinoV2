"""Stage a raw dataset into a flat, sequentially-numbered frame sequence.

Merges every split (train/, val/, ...) under a raw dataset folder and stages
image+label pairs into a destination folder under sequential names
(000001.jpg / 000001.txt, ...) so no trace of the source video filenames is
left. The destination is a single flat frame sequence — no per-video
grouping, no --video tags needed downstream.

    <src>/train/ ┐
                 ├─>  <dst>/000001.jpg, 000001.txt, ...
    <src>/val/   ┘

Only *complete image+label pairs* are staged: an image is staged together
with its sibling YOLO *.txt label, and only if both exist and the label has
at least one annotation. Orphan images (no label), orphan labels (no image)
and empty labels (negative/background frames, 0 boxes) are skipped.

If both a .jpg and .png exist for the same stem, .jpg is preferred and the
collision is reported. The dataset's classes file (classes.txt, or any
*classes*.txt found at the split level or the dataset root) is copied to
<dst>/classes.txt.

Pairs are sorted deterministically (by source video and frame number when
the `<video>_frame_<n>` / `<video>.mkv_<n>` naming is recognizable, else by
filename) before being numbered, so re-runs produce the same mapping.

Copies by default (idempotent, re-runnable, keeps the raw dataset intact);
pass --move to move instead.

Usage:
    python standalone_scripts/prepare_dataset.py \\
        --src "datasets/Doosan Swivel E32-E60" \\
        --dst "datasets/doosan_swivel"
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "datasets" / "Doosan Swivel E32-E60"
DEFAULT_DST = REPO_ROOT / "datasets" / "doosan_swivel"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# "<name>_frame_000205" or "<name>.mp4_1000" / "<name>.mkv_1000"
FRAME_RE = re.compile(r"^(.*?)(?:_frame_(\d+)|\.(?:mp4|mkv)_(\d+))$")


def _is_classes_file(name: str) -> bool:
    return "classes" in name.lower() and name.lower().endswith(".txt")


def _sort_key(stem: str) -> tuple[str, int, str]:
    m = FRAME_RE.match(stem)
    if m:
        frame = int(m.group(2) or m.group(3))
        return (m.group(1), frame, stem)
    return (stem, -1, stem)


def _scan_split(split_dir: Path):
    """Split a source dir into (pairs, orphan_images, orphan_labels, empty_labels, img_conflicts, classes)."""
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    img_conflicts: list[str] = []
    classes: Path | None = None
    for path in split_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt" and _is_classes_file(path.name):
            classes = path
        elif suffix in IMAGE_EXTS:
            if path.stem in images and images[path.stem].suffix.lower() != suffix:
                img_conflicts.append(path.stem)
                if suffix == ".jpg":  # prefer jpg over png
                    images[path.stem] = path
            else:
                images[path.stem] = path
        elif suffix == ".txt":
            labels[path.stem] = path

    pairs = []
    orphan_images = 0
    empty_labels = 0
    for stem, img in images.items():
        lbl = labels.get(stem)
        if lbl is None:
            orphan_images += 1
            continue
        if lbl.stat().st_size == 0 or not lbl.read_text(encoding="utf-8", errors="ignore").strip():
            empty_labels += 1
            continue
        pairs.append((stem, img, lbl))
    orphan_labels = sum(1 for stem in labels if stem not in images)
    return pairs, orphan_images, orphan_labels, empty_labels, img_conflicts, classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help=f"raw dataset folder containing split subdirs (default: {DEFAULT_SRC})")
    ap.add_argument("--dst", default=str(DEFAULT_DST),
                    help=f"destination folder for the staged flat sequence (default: {DEFAULT_DST})")
    ap.add_argument("--splits", default="train,val",
                    help="comma-separated split subdirs to merge (default: train,val)")
    ap.add_argument("--move", action="store_true",
                    help="move files instead of copying (empties the raw splits)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the destination folder before staging")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be staged without touching files")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst = Path(args.dst)
    if not src_root.is_dir():
        ap.error(f"dataset folder not found: {src_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_pairs: list[tuple[str, Path, Path]] = []
    seen_stems: set[str] = set()
    dup_across_splits = 0
    tot_orphan_images = tot_orphan_labels = tot_empty_labels = 0
    all_img_conflicts: list[str] = []
    classes: Path | None = None

    for split in splits:
        split_dir = src_root / split
        if not split_dir.is_dir():
            ap.error(f"split dir not found: {split_dir}")
        pairs, orphan_images, orphan_labels, empty_labels, img_conflicts, split_classes = _scan_split(split_dir)
        tot_orphan_images += orphan_images
        tot_orphan_labels += orphan_labels
        tot_empty_labels += empty_labels
        all_img_conflicts.extend(img_conflicts)
        if split_classes is not None and classes is None:
            classes = split_classes

        kept = 0
        for stem, img, lbl in pairs:
            if stem in seen_stems:
                dup_across_splits += 1
                continue
            seen_stems.add(stem)
            all_pairs.append((stem, img, lbl))
            kept += 1
        print(f"  {split}: pairs={kept}  "
              f"skipped(orphan img={orphan_images}, orphan lbl={orphan_labels}, "
              f"empty lbl={empty_labels}"
              + (f", duplicate stems={len(pairs) - kept}" if len(pairs) - kept else "")
              + ")  jpg/png conflicts=" + str(len(img_conflicts)))

    if classes is None:
        # not found inside any split -> look at the dataset root (e.g. E32_classes.txt)
        root_matches = [p for p in src_root.iterdir() if p.is_file() and _is_classes_file(p.name)]
        if root_matches:
            classes = root_matches[0]

    all_pairs.sort(key=lambda p: _sort_key(p[0]))

    if args.clean and dst.exists() and not args.dry_run:
        shutil.rmtree(dst)
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    verb = "Would stage" if args.dry_run else ("Moving" if args.move else "Copying")
    print(f"{verb} '{src_root.name}' from {src_root}  ->  {dst}  "
          f"({len(all_pairs)} pairs, renamed to sequential 000001.jpg/.txt)")

    if not args.dry_run:
        xfer = shutil.move if args.move else shutil.copy2
        for i, (stem, img, lbl) in enumerate(all_pairs, start=1):
            xfer(str(img), str(dst / f"{i:06d}{img.suffix.lower()}"))
            xfer(str(lbl), str(dst / f"{i:06d}.txt"))
        if classes is not None:
            xfer(str(classes), str(dst / "classes.txt"))

    print(f"\nStaged '{src_root.name}' -> {dst}")
    print(f"total: {len(all_pairs)} complete pairs staged (000001..{len(all_pairs):06d}); "
          f"skipped {tot_orphan_images} orphan images + {tot_orphan_labels} orphan labels + "
          f"{tot_empty_labels} empty labels"
          + (f" + {dup_across_splits} duplicate stems across splits" if dup_across_splits else "")
          + (f"; {len(all_img_conflicts)} jpg/png name conflicts (jpg kept)" if all_img_conflicts else ""))
    if classes is not None:
        print(f"classes.txt {'would be ' if args.dry_run else ''}copied from {classes.name}")
    else:
        print("! no classes file found (looked for *classes*.txt in splits and dataset root)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
