#!/usr/bin/env python3
"""Overlay two YOLO detector checkpoints' predictions on the same video(s), each in its
own color, so they can be visually compared frame-for-frame.

Usage (single video):
    python standalone_scripts/compare_visualize.py \\
        --video "datasets/marmon_station_1/videos/2026-08-06 09-00-50.mkv" \\
        --model-a best=runs/.../weights/best.pt --color-a purple \\
        --model-b marmon_station_1_2=runs/.../weights/marmon_station_1_2.pt --color-b yellow \\
        --out runs/marmon_station_1_inference_sample/compare/2026-08-06_09-00-50_compare.mp4

Usage (many videos -- with no --video/--videos-file given, defaults to the
--top-n-per-day largest-by-filesize videos in --videos-dir for each day found there,
picked fresh from what's on disk at run time):
    python standalone_scripts/compare_visualize.py \\
        --videos-dir datasets/marmon_station_1/videos --top-n-per-day 15 \\
        --model-a best=runs/.../weights/best.pt --color-a purple \\
        --model-b marmon_station_1_2=runs/.../weights/marmon_station_1_2.pt --color-b yellow \\
        --out-dir runs/marmon_station_1_inference_sample/compare

Usage (many videos from a file instead, one line per filename):
    python standalone_scripts/compare_visualize.py \\
        --videos-file datasets/marmon_station_1/selected_45_videos.txt \\
        --videos-dir datasets/marmon_station_1/videos \\
        --model-a best=runs/.../weights/best.pt --color-a purple \\
        --model-b marmon_station_1_2=runs/.../weights/marmon_station_1_2.pt --color-b yellow \\
        --out-dir runs/marmon_station_1_inference_sample/compare

Resumable: each video's output is written to a .tmp file and only atomically renamed
to its final name once fully written, so a killed/interrupted run can just be re-run
with the same command -- videos whose final output already exists are skipped. Pass
--overwrite to force re-processing everything.

TensorRT: pass --engine to export each .pt to a cached TensorRT .engine (next to the
weights, reused on later runs unless --force-export) and run inference off that
instead of plain PyTorch -- typically a large speedup on top of batching. First run
per model takes a few extra minutes to build the engine. Requires the `tensorrt`,
`onnx`, and `onnxslim` packages.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) \d{2}-\d{2}-\d{2}")


def largest_per_day(videos_dir: Path, video_ext: str, top_n: int) -> list[Path]:
    """Group every *.<video_ext> in videos_dir by the date in its filename
    ("YYYY-MM-DD HH-MM-SS.ext") and keep the top_n largest-by-filesize per day --
    larger file size at a fixed duration roughly tracks more motion/activity in
    frame. Computed fresh from what's on disk each time this runs."""
    by_day: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(videos_dir.glob(f"*.{video_ext}")):
        m = _DATE_RE.search(f.name)
        if m:
            by_day[m.group(1)].append(f)

    selected = []
    for day in sorted(by_day):
        top = sorted(by_day[day], key=lambda f: -f.stat().st_size)[:top_n]
        selected.extend(sorted(top))  # chronological within the day, for readability
    return selected


# BGR tuples (OpenCV order)
COLORS = {
    "purple": (240, 32, 160),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "cyan": (255, 255, 0),
    "blue": (255, 0, 0),
    "orange": (0, 140, 255),
    "white": (255, 255, 255),
}


def _color(s: str) -> tuple[int, int, int]:
    if s in COLORS:
        return COLORS[s]
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected a color name ({', '.join(COLORS)}) or 'B,G,R', got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def _parse_model_arg(s: str) -> tuple[str, Path]:
    label, sep, path = s.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"expected label=path/to/weights.pt, got {s!r}")
    return label, Path(path)


def to_engine(weights: Path, device: str, imgsz: int, max_batch: int, half: bool, force: bool) -> Path:
    """Export weights.pt to a TensorRT .engine next to it (cached across runs -- built
    once with dynamic batch shape 1..max_batch, so it keeps working if --batch-size is
    later lowered; raise max_batch and this re-exports). Returns the .engine path."""
    engine_path = weights.with_suffix(".engine")
    if engine_path.exists() and not force:
        print(f"  [engine] reusing cached {engine_path.name}")
        return engine_path

    try:
        import tensorrt  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "TensorRT export requires the 'tensorrt', 'onnx', and 'onnxslim' packages "
            "(pip install tensorrt onnx onnxslim) -- not importable in this environment."
        ) from e

    from ultralytics import YOLO
    print(f"  [engine] exporting {weights.name} -> TensorRT "
          f"(imgsz={imgsz}, batch<={max_batch}, half={half}) ...")
    m = YOLO(str(weights))
    exported = Path(m.export(format="engine", device=device, imgsz=imgsz, half=half,
                              dynamic=True, batch=max_batch))
    if exported != engine_path:
        exported.replace(engine_path)
    print(f"  [engine] -> {engine_path}")
    return engine_path


def draw_boxes(frame, result, color, class_names: list[str] | None, show_labels: bool, thickness: int,
               label_pos: str):
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    h, w = frame.shape[:2]
    for box, c, cf in zip(xyxy, cls, conf):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            name = class_names[int(c)] if class_names and int(c) < len(class_names) else str(int(c))
            text = f"{name} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            if label_pos == "below":
                ty1 = min(y2, h - th - 4)
                ty2 = ty1 + th + 4
            else:
                ty2 = max(y1, th + 4)
                ty1 = ty2 - th - 4
            cv2.rectangle(frame, (x1, ty1), (x1 + tw + 2, ty2), color, -1)
            cv2.putText(frame, text, (x1 + 1, ty2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def process_video(video: Path, out: Path, model_a, model_b, label_a: str, label_b: str,
                   class_names: list[str] | None, args) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # write under a temp name and only rename to `out` once fully written, so a killed
    # run never leaves a partial file sitting at the final name -- that's what makes
    # the existence check in main() a safe "is this one actually done?" test.
    tmp_out = out.with_name(out.name + ".tmp")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(tmp_out), fourcc, fps, (w, h))

    # legend, drawn on every frame (top-left)
    legend = [(label_a, args.color_a), (label_b, args.color_b)]

    show_labels = not args.no_labels
    frames, n = [], 0
    t0 = time.time()

    def flush():
        nonlocal frames, n
        if not frames:
            return
        results_a = model_a.predict(frames, verbose=False, device=args.device, save=False,
                                     conf=args.conf, iou=args.iou, max_det=args.max_det)
        results_b = model_b.predict(frames, verbose=False, device=args.device, save=False,
                                     conf=args.conf, iou=args.iou, max_det=args.max_det)
        for frame, ra, rb in zip(frames, results_a, results_b):
            out_frame = frame.copy()
            draw_boxes(out_frame, ra, args.color_a, class_names, show_labels, args.thickness, args.label_pos_a)
            draw_boxes(out_frame, rb, args.color_b, class_names, show_labels, args.thickness, args.label_pos_b)
            y = 20
            for name, color in legend:
                cv2.putText(out_frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                y += 22
            vw.write(out_frame)
            n += 1
        frames = []

    while True:
        if args.limit_frames > 0 and n >= args.limit_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == args.batch_size:
            flush()
    flush()

    cap.release()
    vw.release()
    tmp_out.replace(out)
    elapsed = time.time() - t0
    print(f"  {n} frames in {elapsed:.1f}s ({n / elapsed:.1f} fps) -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[], help="repeatable; a single video to process")
    ap.add_argument("--videos-file", default=None,
                     help="text file, one video filename/path per line, resolved against --videos-dir. "
                          "If neither this nor --video is given, falls back to the --top-n-per-day "
                          "largest-by-filesize videos per day found in --videos-dir")
    ap.add_argument("--videos-dir", default="datasets/marmon_station_1/videos",
                     help="base dir for relative entries in --videos-file, and for the "
                          "largest-per-day fallback selection")
    ap.add_argument("--video-ext", default="mkv", help="extension to glob for the largest-per-day fallback")
    ap.add_argument("--top-n-per-day", type=int, default=15,
                     help="how many largest-by-filesize videos to keep per day for the fallback selection")
    ap.add_argument("--model-a", required=True, type=_parse_model_arg, help="label=path/to/weights.pt")
    ap.add_argument("--color-a", default="purple", type=_color)
    ap.add_argument("--model-b", required=True, type=_parse_model_arg, help="label=path/to/weights.pt")
    ap.add_argument("--color-b", default="yellow", type=_color)
    ap.add_argument("--classes", default="datasets/marmon_station_1/classes.txt")
    ap.add_argument("--out", default=None, help="output path; only valid with a single --video")
    ap.add_argument("--out-dir", default="runs/compare_visualize",
                     help="output dir when processing multiple videos (or --video without --out)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--label-pos-a", choices=["above", "below"], default="below",
                     help="where to draw model-a's class/conf text relative to its box")
    ap.add_argument("--label-pos-b", choices=["above", "below"], default="above",
                     help="where to draw model-b's class/conf text relative to its box")
    ap.add_argument("--no-labels", action="store_true", help="don't draw class name/conf text, just boxes")
    ap.add_argument("--limit-frames", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true",
                     help="re-process videos whose output already exists (default: skip them, resumable)")
    ap.add_argument("--engine", action="store_true",
                     help="export/use TensorRT .engine instead of the .pt for both models (see module docstring)")
    ap.add_argument("--imgsz", type=int, default=640, help="export/inference image size for --engine")
    ap.add_argument("--engine-fp32", action="store_true",
                     help="build the TensorRT engine in fp32 instead of the default fp16")
    ap.add_argument("--force-export", action="store_true",
                     help="rebuild the .engine even if a cached one already exists")
    args = ap.parse_args()

    from ultralytics import YOLO

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_absolute():
        videos_dir = REPO_ROOT / videos_dir

    videos: list[Path] = []
    for v in args.video:
        p = Path(v)
        videos.append(p if p.is_absolute() else REPO_ROOT / p)
    if args.videos_file:
        vf = Path(args.videos_file)
        if not vf.is_absolute():
            vf = REPO_ROOT / vf
        for line in vf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            videos.append(p if p.is_absolute() else videos_dir / p)
    if not videos:
        videos = largest_per_day(videos_dir, args.video_ext, args.top_n_per_day)
        print(f"No --video/--videos-file given: picked {len(videos)} largest-by-filesize "
              f"videos ({args.top_n_per_day}/day) from {videos_dir}")
    for v in videos:
        if not v.exists():
            raise FileNotFoundError(v)
    if args.out and len(videos) > 1:
        raise ValueError("--out only makes sense with a single video; use --out-dir for multiple")

    (label_a, weights_a), (label_b, weights_b) = args.model_a, args.model_b
    weights_a = weights_a if weights_a.is_absolute() else REPO_ROOT / weights_a
    weights_b = weights_b if weights_b.is_absolute() else REPO_ROOT / weights_b

    class_names = None
    classes_path = Path(args.classes)
    if not classes_path.is_absolute():
        classes_path = REPO_ROOT / classes_path
    if classes_path.exists():
        class_names = [l.strip() for l in classes_path.read_text().splitlines() if l.strip()]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # resolve outputs up front so we know what's already done before paying for model
    # load (and a TensorRT export, if requested) -- resumability shouldn't cost anything
    # extra on a run that turns out to have nothing left to do.
    jobs = []  # (video, out)
    skipped_done = 0
    for video in videos:
        out = Path(args.out) if args.out else out_dir / f"{video.stem}_compare.mp4"
        if not out.is_absolute():
            out = REPO_ROOT / out
        if out.exists() and not args.overwrite:
            skipped_done += 1
            continue
        jobs.append((video, out))

    if skipped_done:
        print(f"{skipped_done}/{len(videos)} video(s) already done, skipping (pass --overwrite to redo them)")
    if not jobs:
        print("Nothing to do.")
        return 0

    if args.engine:
        weights_a = to_engine(weights_a, args.device, args.imgsz, args.batch_size,
                               half=not args.engine_fp32, force=args.force_export)
        weights_b = to_engine(weights_b, args.device, args.imgsz, args.batch_size,
                               half=not args.engine_fp32, force=args.force_export)

    print(f"[a] loading {label_a} ({weights_a.name}, color={args.color_a}) ...")
    model_a = YOLO(str(weights_a))
    print(f"[b] loading {label_b} ({weights_b.name}, color={args.color_b}) ...")
    model_b = YOLO(str(weights_b))

    t_start = time.time()
    for i, (video, out) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {video.name}")
        process_video(video, out, model_a, model_b, label_a, label_b, class_names, args)

    print(f"\nDone: {len(jobs)} video(s) in {time.time() - t_start:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
