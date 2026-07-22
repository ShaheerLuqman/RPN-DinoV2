#!/usr/bin/env python3
"""Phase-1 recall evaluation for the single-class 'object' detector.

Localization is class-agnostic, but we score recall broken out by the ORIGINAL
class (from gas_valve_2view labels) so we can see which real objects get found —
the same overall-vs-fair-average lens used in Phase 2. A GT box counts as
"found" if any prediction overlaps it at IoU >= threshold.

Reports, on the 1000-frame Phase-2 val split:
  - micro recall (every box equal) and macro recall (every class equal) @IoU .5
  - Average Recall (AR) = mean recall over IoU .50:.05:.95
  - per-class recall and miss counts
  - proposals per frame (review-load proxy)
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR

DATA = Path("/home/retrocausal-train/Documents/RPN+DinoV2/datasets/gas_valve_2view")
VAL_IMG = Path("/home/retrocausal-train/Documents/RPN+DinoV2/phase1/dataset/images/val")
CLASSES = DATA / "classes.txt"
IOUS = np.arange(0.5, 1.0, 0.05)


def load_classes(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def gt_boxes_px(stem, W, H):
    """original-class GT boxes as (cls, x1,y1,x2,y2) in pixels."""
    out = []
    for line in (DATA / f"{stem}.txt").read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0])); xc, yc, w, h = map(float, p[1:])
        x1 = (xc - w / 2) * W; y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W; y2 = (yc + h / 2) * H
        out.append((c, x1, y1, x2, y2))
    return out


def iou_matrix(gt, pr):
    """gt: (G,4) pr: (P,4) -> (G,P) IoU."""
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)), np.float32)
    g = gt[:, None, :]; p = pr[None, :, :]
    ix1 = np.maximum(g[..., 0], p[..., 0]); iy1 = np.maximum(g[..., 1], p[..., 1])
    ix2 = np.minimum(g[..., 2], p[..., 2]); iy2 = np.minimum(g[..., 3], p[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    union = ag[:, None] + ap[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val_img", default=str(VAL_IMG), help="dir of val images to score")
    ap.add_argument("--conf", type=float, default=0.001)   # recall-first: keep everything
    ap.add_argument("--iou_nms", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--rtdetr", action="store_true")
    ap.add_argument("--out", default="/home/retrocausal-train/Documents/RPN+DinoV2/phase1/recall_report.txt")
    args = ap.parse_args()

    names = load_classes(CLASSES)
    Model = RTDETR if args.rtdetr else YOLO
    model = Model(args.weights)

    val_img = Path(args.val_img)
    stems = sorted(p.stem for p in val_img.glob("*.png"))
    nC = len(names)
    # per-class: gt count, and #found at each IoU threshold
    gt_n = np.zeros(nC, int)
    found = np.zeros((nC, len(IOUS)), int)
    n_pred = 0

    for i in range(0, len(stems), 64):
        batch = stems[i:i + 64]
        paths = [str(val_img / f"{s}.png") for s in batch]
        res = model.predict(paths, conf=args.conf, iou=args.iou_nms, imgsz=args.imgsz,
                            max_det=args.max_det, device=args.device, verbose=False)
        for s, r in zip(batch, res):
            H, W = r.orig_shape
            pr = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
            n_pred += len(pr)
            gt = gt_boxes_px(s, W, H)
            if not gt:
                continue
            gcls = np.array([g[0] for g in gt])
            gbox = np.array([g[1:] for g in gt], np.float32)
            M = iou_matrix(gbox, pr)               # (G,P)
            best = M.max(axis=1) if pr.shape[0] else np.zeros(len(gt))
            for c in np.unique(gcls):
                m = gcls == c
                gt_n[c] += int(m.sum())
                for ti, thr in enumerate(IOUS):
                    found[c, ti] += int((best[m] >= thr).sum())

    present = [c for c in range(nC) if gt_n[c] > 0]
    rec50 = np.array([found[c, 0] / gt_n[c] for c in present])
    ar = np.array([found[c].sum() / (gt_n[c] * len(IOUS)) for c in present])  # per-class AR
    micro50 = found[present, 0].sum() / gt_n[present].sum()
    micro_ar = found[present, :].sum() / (gt_n[present].sum() * len(IOUS))

    order = sorted(present, key=lambda c: found[c, 0] / gt_n[c])
    lines = [f"=== PHASE-1 RECALL ({Path(args.weights).name}) ===",
             f"val frames: {len(stems)}   predictions: {n_pred}   "
             f"proposals/frame: {n_pred/len(stems):.1f}   conf>={args.conf}",
             f"micro recall@.5 : {micro50:.4f}  (every box equal)",
             f"macro recall@.5 : {rec50.mean():.4f}  (every class equal)",
             f"micro AR .5:.95 : {micro_ar:.4f}",
             f"macro AR .5:.95 : {ar.mean():.4f}",
             "",
             f"{'id':>3} {'class':<28} {'gt':>6} {'rec@.5':>7} {'AR':>6} {'missed':>7}",
             "-" * 62]
    for c in order:
        r5 = found[c, 0] / gt_n[c]
        a = found[c].sum() / (gt_n[c] * len(IOUS))
        lines.append(f"{c:>3} {names[c][:28]:<28} {gt_n[c]:>6} {r5:>7.3f} {a:>6.3f} "
                     f"{gt_n[c]-found[c,0]:>7}")
    report = "\n".join(lines)
    print(report)
    Path(args.out).write_text(report)
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
