#!/usr/bin/env python3
"""Stacked pred-vs-GT visualization video.

Top panel  = Phase1->Phase2 PREDICTIONS, red boxes + predicted class tag.
Bottom panel = GROUND TRUTH, green boxes + class tag.
Reads the per-frame predictions saved by phase1_phase2_e2e.py (no model needed).
"""
import argparse, json
from pathlib import Path
import cv2

ROOT = Path("/home/retrocausal-train/Documents/RPN+DinoV2")
VAL_IMG = ROOT / "phase1/dataset/images/val"
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)


def draw_boxes(img, boxes, color, label_fn, fs=0.4):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lab = label_fn(b)
        if not lab:
            continue
        (tw, th), _ = cv2.getTextSize(lab, FONT, fs, 1)
        ytop = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, ytop), (x1 + tw + 3, ytop + th + 4), color, -1)
        cv2.putText(img, lab, (x1 + 1, ytop + th + 1), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def banner(img, text, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 12), color, -1)
    cv2.putText(img, text, (8, th + 6), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", default=str(ROOT / "phase1/e2e_predictions.json"))
    ap.add_argument("--out", default=str(ROOT / "phase1/e2e_pred_vs_gt.mp4"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=500, help="-1 = all val frames")
    ap.add_argument("--show_conf", action="store_true", help="append name-confidence to pred tags")
    args = ap.parse_args()

    data = json.loads(Path(args.pred_json).read_text())
    stems = sorted(data.keys())
    if args.limit > 0:
        stems = stems[:args.limit]

    d0 = data[stems[0]]
    W, H = d0["W"], d0["H"]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, 2 * H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open (codec/path issue)")

    pred_lab = (lambda b: f'{b["name"]} {b["name_conf"]:.2f}') if args.show_conf else (lambda b: b["name"])
    for i, s in enumerate(stems):
        im = cv2.imread(str(VAL_IMG / f"{s}.png"))
        if im is None:
            continue
        top, bot = im.copy(), im.copy()
        draw_boxes(top, data[s]["pred"], RED, pred_lab)
        draw_boxes(bot, data[s]["gt"], GREEN, lambda b: b["name"])
        banner(top, f"PREDICTED  ({s})", RED)
        banner(bot, "GROUND TRUTH", GREEN)
        stacked = cv2.vconcat([top, bot])
        cv2.line(stacked, (0, H), (W, H), (255, 255, 255), 1)
        writer.write(stacked)
        if i % 100 == 0:
            print(f"  {i}/{len(stems)} frames", flush=True)
    writer.release()
    print(f"wrote {args.out}  ({len(stems)} frames @ {args.fps} fps, {W}x{2*H})")


if __name__ == "__main__":
    main()
