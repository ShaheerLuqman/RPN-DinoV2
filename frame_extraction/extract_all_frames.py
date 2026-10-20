#!/usr/bin/env python3
"""Dump every native frame of a folder of videos to individual .jpg files — no
filtering, no selection. This is the "all frames" side of a before/after preview
against frame_extraction/select_frames.py's diverse subset (see
docs/frame_selection.md): put this script's output folder next to a
select_frames.py `selected/` folder and compare the two directly.

Frames are named `<video-stem>_frame_NNNNNN.jpg` — the same convention
prepare_dataset.py / select_frames.py already recognize (FRAME_RE), so the output
folder can be treated as any other per-video-named frame folder by the rest of this
repo's tooling if needed.

No GPU, no model — this is a plain decode-and-write loop, safe to run alongside a
select_frames.py run or a training job without any GPU contention.

Usage:
    python frame_extraction/extract_all_frames.py \\
        --videos-dir datasets/marmon_station_1/videos \\
        --out-dir runs/frame_selection/marmon_station_1/all_frames
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
JPEG_QUALITY = 95  # matches prepare_dataset.py's JPEG_QUALITY


def extract_video(video: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{video.stem}_frame_{n:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n += 1
    cap.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--video-ext", default="mkv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=-1, help="only process the first N videos (-1 = all)")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    videos_dir = videos_dir if videos_dir.is_absolute() else REPO_ROOT / videos_dir
    videos = sorted(videos_dir.glob(f"*.{args.video_ext}"))
    if args.limit > 0:
        videos = videos[:args.limit]
    if not videos:
        raise FileNotFoundError(f"no *.{args.video_ext} files under {videos_dir}")

    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting every frame of {len(videos)} video(s) from {videos_dir} -> {out_dir}")
    t0 = time.time()
    total = 0
    for i, video in enumerate(videos, 1):
        n = extract_video(video, out_dir)
        total += n
        elapsed = time.time() - t0
        print(f"  ({i}/{len(videos)}) {video.name}: {n} frames  "
              f"[{total} total so far, {elapsed:.0f}s elapsed]", flush=True)

    print(f"\nDone: {total} frames from {len(videos)} video(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
