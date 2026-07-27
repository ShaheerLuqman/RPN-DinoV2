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
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)
PURPLE = (226, 43, 138)   # BGR blue-violet — the SAM segment fill


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


def _class_colors(n: int):
    """Return `n` visually distinct BGR colors, one per class id (evenly spaced hues,
    deterministic so a class keeps its color across frames)."""
    import numpy as _np
    n = max(n, 1)
    hues = (_np.arange(n) * 180.0 / n).astype(_np.uint8)
    hsv = _np.stack([hues, _np.full(n, 220, _np.uint8), _np.full(n, 255, _np.uint8)], 1)
    bgr = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in row) for row in bgr]


def _draw_box(img, box, color, label, fs=0.4):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
    yt = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, yt), (x1 + tw + 3, yt + th + 4), color, -1)
    cv2.putText(img, label, (x1 + 1, yt + th + 1), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)


def _overlay_mask(img, mask, color, alpha=0.45):
    """Blend `color` into `img` wherever `mask` is set, keeping the object visible."""
    H, W = img.shape[:2]
    if mask.shape[:2] != (H, W):                       # SAM may return at model res
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    sel = mask.astype(bool)
    if not sel.any():
        return
    img[sel] = (img[sel] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def render_mask_video(cfg, masker):
    """Standalone SAM-mask video: for each frame, prompt SAM with the *predicted*
    boxes (from predictions.json), fill each segment with its class color (translucent)
    and outline the box in the same color. Written to `<work>_masks.mp4` — separate
    from the pred-vs-GT comparison video."""
    v = cfg.visualize
    data = json.loads(cfg.predictions_json.read_text())
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    colors = _class_colors(len(cfg.classes))
    stems = sorted(data.keys())
    if getattr(v, "limit", -1) and v.limit > 0:
        stems = stems[:v.limit]

    d0 = data[stems[0]]
    W, H = d0["W"], d0["H"]
    out = cfg.work / f"{cfg.work.name}_masks.mp4"
    fps = int(v.fps)
    crf = int(getattr(v, "crf", 28))

    proc = _open_ffmpeg(out, W, H, fps, crf)
    writer = None
    if proc is None:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError("no ffmpeg and OpenCV VideoWriter failed to open")

    written = n_masks = 0
    for i, s in enumerate(stems):
        im = cv2.imread(str(images_dir / f"{s}.{ext}"))
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        preds = data[s]["pred"]
        boxes = [p["box"] for p in preds]
        masks = masker.masks_for(str(images_dir / f"{s}.{ext}"), boxes) if boxes else []
        for p, m in zip(preds, masks):
            col = colors[int(p["cls"]) % len(colors)]
            if m is not None:
                _overlay_mask(im, m, col)
                n_masks += 1
            _draw_box(im, p["box"], col, p["name"])
        _banner(im, f"SAM MASKS  ({s})", (60, 60, 60))
        if proc is not None:
            proc.stdin.write(im.tobytes())
        else:
            writer.write(im)
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
    print(f"wrote {out}  ({written} frames, {n_masks} masks @ {fps} fps, "
          f"{W}x{H}, {codec}, {size_mb:.1f} MB)")
    return out


# Distinct colors for the candidate masks *within* one box (whole/part/subpart).
MASK_PALETTE = [(226, 43, 138),   # blue-violet
                (0, 215, 255),     # gold
                (0, 255, 128),     # spring green
                (255, 128, 0),     # azure
                (203, 0, 255)]     # magenta


def _mask_at_frame(mask, H, W):
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def render_multimask_video(cfg, masker):
    """Standalone SAM multimask video: for each *predicted* box, ask SAM for its
    whole/part/subpart candidate masks and draw each in a distinct color (translucent
    fill + solid contour). The box label carries the candidate count, so composite
    classes that decompose into several objects stand out. Written to
    `<work>_masks_multi.mp4`."""
    v = cfg.visualize
    data = json.loads(cfg.predictions_json.read_text())
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    stems = sorted(data.keys())
    if getattr(v, "limit", -1) and v.limit > 0:
        stems = stems[:v.limit]

    d0 = data[stems[0]]
    W, H = d0["W"], d0["H"]
    out = cfg.work / f"{cfg.work.name}_masks_multi.mp4"
    fps = int(v.fps)
    crf = int(getattr(v, "crf", 28))

    proc = _open_ffmpeg(out, W, H, fps, crf)
    writer = None
    if proc is None:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError("no ffmpeg and OpenCV VideoWriter failed to open")

    written = n_masks = n_multi = 0
    for i, s in enumerate(stems):
        im = cv2.imread(str(images_dir / f"{s}.{ext}"))
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        preds = data[s]["pred"]
        boxes = [p["box"] for p in preds]
        per_box = masker.multi_masks_for(str(images_dir / f"{s}.{ext}"), boxes) if boxes else []
        for p, cands in zip(preds, per_box):
            n_masks += len(cands)
            if len(cands) > 1:
                n_multi += 1
            # largest first, so smaller parts/subparts draw on top and stay visible
            for j, (m, sc) in enumerate(sorted(cands, key=lambda ms: -int(ms[0].sum()))):
                col = MASK_PALETTE[j % len(MASK_PALETTE)]
                mm = _mask_at_frame(m, H, W)
                _overlay_mask(im, mm, col, alpha=0.35)
                cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnts, -1, col, 2)
            _draw_box(im, p["box"], (255, 255, 255), f'{p["name"]} x{len(cands)}')
        _banner(im, f"SAM MULTIMASK  ({s})", (60, 60, 60))
        if proc is not None:
            proc.stdin.write(im.tobytes())
        else:
            writer.write(im)
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
    print(f"wrote {out}  ({written} frames, {n_masks} masks, {n_multi} boxes with >1 mask "
          f"@ {fps} fps, {W}x{H}, {codec}, {size_mb:.1f} MB)")
    return out
