#!/usr/bin/env python3
"""Run YOLO detector inference over a folder of videos, saving per-frame YOLO-format
labels and an annotated (boxes drawn) copy of each video.

Frames are read from each video and pushed through the model in batches (this alone
gives a large GPU-throughput win over frame-by-frame streaming). Annotated-frame
drawing + video/label writing happens on a background thread so it overlaps with the
next batch's GPU inference instead of serializing after it.

Usage:
    python standalone_scripts/infer_videos.py \\
        --videos-dir datasets/marmon_station_1/videos \\
        --model best=runs/20260811_122814_marmon_1_1000_yolo/detector/train/weights/best.pt \\
        --classes datasets/marmon_station_1/classes.txt \\
        --out-dir runs/marmon_station_1_inference \\
        --limit 1
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_model_arg(s: str) -> tuple[str, Path]:
    label, sep, path = s.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"expected label=path/to/weights.pt, got {s!r}")
    return label, Path(path)


def _writer_worker(q: "queue.Queue", vw: cv2.VideoWriter | None, labels_dir: Path | None,
                    save_video: bool, save_labels: bool) -> None:
    idx = 0
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        result = item
        if save_video:
            vw.write(result.plot())
        if save_labels:
            lines = []
            if result.boxes is not None and len(result.boxes):
                xywhn = result.boxes.xywhn.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                for c, box, cf in zip(cls, xywhn, conf):
                    lines.append(f"{int(c)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {cf:.4f}\n")
            (labels_dir / f"frame_{idx:06d}.txt").write_text("".join(lines))
        idx += 1
        q.task_done()


def infer_video(model, video: Path, out_dir: Path, batch_size: int, conf: float, iou: float,
                 max_det: int, device: str, save_video: bool, save_labels: bool) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_out_dir = out_dir / video.stem
    vid_out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = vid_out_dir / "labels"
    if save_labels:
        labels_dir.mkdir(exist_ok=True)

    vw = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(vid_out_dir / "annotated.mp4"), fourcc, fps, (w, h))

    q: "queue.Queue" = queue.Queue(maxsize=4 * batch_size)
    wt = threading.Thread(target=_writer_worker, args=(q, vw, labels_dir, save_video, save_labels), daemon=True)
    wt.start()

    t0 = time.time()
    frames, n = [], 0

    def flush():
        nonlocal frames
        if not frames:
            return
        results = model.predict(frames, verbose=False, device=device, save=False,
                                 conf=conf, iou=iou, max_det=max_det)
        for r in results:
            q.put(r)
        frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        n += 1
        if len(frames) == batch_size:
            flush()
    flush()
    q.put(None)
    q.join()
    wt.join()
    t1 = time.time()

    cap.release()
    if vw is not None:
        vw.release()

    elapsed = t1 - t0
    return {
        "video": video.name, "frames": n, "expected_frames": total,
        "elapsed_s": elapsed, "fps": n / elapsed if elapsed else 0.0,
        "out_dir": str(vid_out_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", default="datasets/marmon_station_1/videos")
    ap.add_argument("--video-ext", default="mkv")
    ap.add_argument("--model", action="append", type=_parse_model_arg, required=True,
                     help="label=path/to/weights.pt, repeatable")
    ap.add_argument("--out-dir", default="runs/marmon_station_1_inference")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=-1, help="only process the first N videos (-1 = all)")
    ap.add_argument("--no-video", action="store_true", help="skip annotated-video output, labels only")
    ap.add_argument("--no-labels", action="store_true", help="skip label .txt output, video only")
    args = ap.parse_args()

    from ultralytics import YOLO

    videos_dir = REPO_ROOT / args.videos_dir if not Path(args.videos_dir).is_absolute() else Path(args.videos_dir)
    videos = sorted(videos_dir.glob(f"*.{args.video_ext}"))
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"no *.{args.video_ext} files under {videos_dir}")

    out_root = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    save_video = not args.no_video
    save_labels = not args.no_labels

    for label, weights in args.model:
        weights = weights if weights.is_absolute() else (REPO_ROOT / weights)
        if not weights.exists():
            raise FileNotFoundError(weights)
        print(f"\n[{label}] loading {weights} ...")
        model = YOLO(str(weights))
        model_out = out_root / label

        for i, video in enumerate(videos, 1):
            print(f"[{label}] ({i}/{len(videos)}) {video.name} ...", end=" ", flush=True)
            stats = infer_video(model, video, model_out, args.batch_size, args.conf, args.iou,
                                 args.max_det, args.device, save_video, save_labels)
            print(f"{stats['frames']} frames in {stats['elapsed_s']:.1f}s "
                  f"({stats['fps']:.1f} fps) -> {stats['out_dir']}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
