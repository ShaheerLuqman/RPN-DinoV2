"""Stacked pred-vs-GT comparison video from a predictions.json.

Top = predictions (red boxes + predicted class tag).
Bottom = ground truth (green boxes + class tag).

Frames are piped into ffmpeg (H.264, CRF) for a small file; if ffmpeg is not on
PATH it falls back to OpenCV's (bulkier) mp4v writer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)


def _draw(img, boxes, color, label_fn, fs=0.4):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lab = label_fn(b)
        if not lab:
            continue
        (tw, th), _ = cv2.getTextSize(lab, FONT, fs, 1)
        yt = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, yt), (x1 + tw + 3, yt + th + 4), color, -1)
        cv2.putText(img, lab, (x1 + 1, yt + th + 1), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)


def _banner(img, text, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 12), color, -1)
    cv2.putText(img, text, (8, th + 6), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _open_ffmpeg(out: Path, W: int, H: int, fps: int, crf: int):
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_video(cfg, predictions: dict | None = None):
    v = cfg.visualize
    data = predictions or json.loads(cfg.predictions_json.read_text())
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    stems = sorted(data.keys())
    if getattr(v, "limit", -1) and v.limit > 0:
        stems = stems[:v.limit]

    d0 = data[stems[0]]
    W, H = d0["W"], d0["H"]
    out = cfg.work / f"{cfg.work.name}_viz.mp4"
    fps = int(v.fps)
    crf = int(getattr(v, "crf", 28))

    proc = _open_ffmpeg(out, W, 2 * H, fps, crf)
    writer = None
    if proc is None:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, 2 * H))
        if not writer.isOpened():
            raise RuntimeError("no ffmpeg and OpenCV VideoWriter failed to open")

    show_conf = getattr(v, "show_conf", False)
    pred_lab = ((lambda b: f'{b["name"]} {b["name_conf"]:.2f}') if show_conf
                else (lambda b: b["name"]))

    written = 0
    for i, s in enumerate(stems):
        im = cv2.imread(str(images_dir / f"{s}.{ext}"))
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        top, bot = im.copy(), im.copy()
        _draw(top, data[s]["pred"], RED, pred_lab)
        _draw(bot, data[s]["gt"], GREEN, lambda b: b["name"])
        _banner(top, f"PREDICTED  ({s})", RED)
        _banner(bot, "GROUND TRUTH", GREEN)
        frame = cv2.vconcat([top, bot])
        cv2.line(frame, (0, H), (W, H), (255, 255, 255), 1)
        if proc is not None:
            proc.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
        written += 1
        if i % 500 == 0:
            print(f"  {i}/{len(stems)} frames", flush=True)

    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
        codec = f"H.264 crf {crf}"
    else:
        writer.release()
        codec = "mp4v"
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({written} frames @ {fps} fps, {W}x{2 * H}, {codec}, {size_mb:.1f} MB)")
    return out
