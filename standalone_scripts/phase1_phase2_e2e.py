#!/usr/bin/env python3
"""End-to-end Phase-1 -> Phase-2: detect boxes, then name each with DINOv2 kNN.

Phase 1 (YOLO single-class) proposes boxes on the held-out val frames; each box is
cropped, embedded with DINOv2 (same config as Phase 2), and named by kNN vote against
the all-frames reference (the ref_all pool = every non-val GT crop). We then match
predictions to GT (IoU>=0.5) and score:

  detection recall      : GT boxes that got a proposal
  naming acc | found    : of found boxes, fraction named correctly (compare to the
                          Phase-2 GT-box upper bound: ~97.8% micro / ~89.1% macro)
  end-to-end            : GT boxes both found AND named correctly (recall x naming)

Also saves per-frame predictions to JSON for the visualization.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `phase2_eval` is importable
from phase2_eval import load_classes, embed, predict_knn

ROOT = Path("/home/retrocausal-train/Documents/RPN+DinoV2")
DATA = ROOT / "datasets/gas_valve_2view"
VAL_IMG = ROOT / "phase1/dataset/images/val"
CLASSES = DATA / "classes.txt"
REF_CACHE = ROOT / "phase2_results/ref_all/cache_even_seed0_refall_min20_q1000.npz"
IOUS = np.arange(0.5, 1.0, 0.05)


def gt_boxes_px(stem, W, H):
    out = []
    for line in (DATA / f"{stem}.txt").read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0])); xc, yc, w, h = map(float, p[1:])
        out.append((c, (xc - w/2)*W, (yc - h/2)*H, (xc + w/2)*W, (yc + h/2)*H))
    return out


def iou_matrix(gt, pr):
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)), np.float32)
    g = gt[:, None, :]; p = pr[None, :, :]
    ix1 = np.maximum(g[...,0], p[...,0]); iy1 = np.maximum(g[...,1], p[...,1])
    ix2 = np.minimum(g[...,2], p[...,2]); iy2 = np.minimum(g[...,3], p[...,3])
    iw = np.clip(ix2-ix1, 0, None); ih = np.clip(iy2-iy1, 0, None); inter = iw*ih
    ag = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]); ap = (pr[:,2]-pr[:,0])*(pr[:,3]-pr[:,1])
    return inter / np.clip(ag[:,None] + ap[None,:] - inter, 1e-9, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT/"phase1/runs/yolo11l_obj-4/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.1)      # clean operating point
    ap.add_argument("--iou_nms", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=-1, help="only first N val frames (debug)")
    ap.add_argument("--pred_json", default=str(ROOT/"phase1/e2e_predictions.json"))
    ap.add_argument("--out", default=str(ROOT/"phase1/e2e_report.txt"))
    args = ap.parse_args()

    names = load_classes(CLASSES); nC = len(names)
    det = YOLO(args.weights)

    print("loading reference embeddings ...", flush=True)
    d = np.load(REF_CACHE); Xr, yr = d["Xr"], d["yr"]
    ref_classes = set(yr.tolist())

    dev = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    emb_model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev)

    stems = sorted(p.stem for p in VAL_IMG.glob("*.png"))
    if args.limit > 0:
        stems = stems[:args.limit]

    # ---- pass 1: detect + collect crops ----
    crops, meta = [], []          # meta: (stem_idx, x1,y1,x2,y2, det_conf)
    per_frame_boxes = {}          # stem -> list of (x1,y1,x2,y2,det_conf, crop_idx or -1)
    print(f"detecting on {len(stems)} val frames ...", flush=True)
    for i in range(0, len(stems), 64):
        batch = stems[i:i+64]
        paths = [str(VAL_IMG/f"{s}.png") for s in batch]
        res = det.predict(paths, conf=args.conf, iou=args.iou_nms, imgsz=args.imgsz,
                          max_det=args.max_det, device=args.device, verbose=False)
        for s, r in zip(batch, res):
            img = r.orig_img                      # BGR
            H, W = r.orig_shape
            boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0,4))
            confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
            fb = []
            for (x1,y1,x2,y2), cf in zip(boxes, confs):
                xi1,yi1 = max(0,int(round(x1))), max(0,int(round(y1)))
                xi2,yi2 = min(W,int(round(x2))), min(H,int(round(y2)))
                ci = -1
                if xi2-xi1 >= 4 and yi2-yi1 >= 4:
                    crop = cv2.cvtColor(img[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2RGB)
                    ci = len(crops)
                    crops.append(Image.fromarray(crop))
                    meta.append((s, float(x1),float(y1),float(x2),float(y2), float(cf)))
                fb.append([float(x1),float(y1),float(x2),float(y2), float(cf), ci])
            per_frame_boxes[s] = fb
        if (i//64) % 4 == 0:
            print(f"  {min(i+64,len(stems))}/{len(stems)} frames, {len(crops)} boxes", flush=True)

    # ---- embed + name all predicted crops ----
    print(f"embedding {len(crops)} predicted crops ...", flush=True)
    Xq = embed(crops, proc, emb_model, dev, batch=64)
    print("kNN naming ...", flush=True)
    pred_cls, name_conf = predict_knn(Xq, Xr, yr, k=args.knn_k)

    # ---- score + build per-frame predictions ----
    gt_n = np.zeros(nC, int); found = np.zeros(nC, int); named_ok = np.zeros(nC, int)
    fp_total = 0
    pred_json = {}
    for s in stems:
        im = cv2.imread(str(VAL_IMG/f"{s}.png")); H, W = im.shape[:2]
        gt = gt_boxes_px(s, W, H)
        fb = per_frame_boxes[s]
        pr_boxes = np.array([[b[0],b[1],b[2],b[3]] for b in fb], np.float32) if fb else np.zeros((0,4),np.float32)
        pr_cls = np.array([int(pred_cls[b[5]]) if b[5] >= 0 else -1 for b in fb])
        pr_nconf = np.array([float(name_conf[b[5]]) if b[5] >= 0 else 0.0 for b in fb])

        matched_pred = np.zeros(len(fb), bool)
        if gt:
            gcls = np.array([g[0] for g in gt])
            gbox = np.array([g[1:] for g in gt], np.float32)
            M = iou_matrix(gbox, pr_boxes)                 # (G,P)
            for gi, g in enumerate(gt):
                c = gcls[gi]; gt_n[c] += 1
                if pr_boxes.shape[0] == 0:
                    continue
                pj = int(M[gi].argmax())
                if M[gi, pj] >= 0.5:
                    found[c] += 1
                    matched_pred[pj] = True
                    if pr_cls[pj] == c:
                        named_ok[c] += 1
        fp_total += int((~matched_pred).sum())

        pred_json[s] = {
            "W": W, "H": H,
            "pred": [{"box":[round(b[0],1),round(b[1],1),round(b[2],1),round(b[3],1)],
                      "cls": int(pr_cls[k]), "name": names[pr_cls[k]] if pr_cls[k]>=0 else "?",
                      "det_conf": round(b[4],3), "name_conf": round(float(pr_nconf[k]),3)}
                     for k, b in enumerate(fb)],
            "gt": [{"box":[round(g[1],1),round(g[2],1),round(g[3],1),round(g[4],1)],
                    "cls": g[0], "name": names[g[0]]} for g in gt],
        }

    Path(args.pred_json).write_text(json.dumps(pred_json))

    present = [c for c in range(nC) if gt_n[c] > 0]
    def micro(a, b): return a[present].sum() / max(b[present].sum(), 1)
    rec_mi = micro(found, gt_n)
    name_mi = found[present].sum() and named_ok[present].sum()/found[present].sum()
    e2e_mi = named_ok[present].sum()/gt_n[present].sum()
    rec_ma = np.mean([found[c]/gt_n[c] for c in present])
    name_ma = np.mean([named_ok[c]/found[c] for c in present if found[c] > 0])
    e2e_ma = np.mean([named_ok[c]/gt_n[c] for c in present])

    lines = [f"=== END-TO-END Phase1->Phase2 ({Path(args.weights).name}, conf {args.conf}) ===",
             f"val frames: {len(stems)}   predicted boxes: {len(meta)}   "
             f"false positives (no GT match): {fp_total}",
             f"reference: ref_all ({len(yr)} crops), kNN k={args.knn_k}",
             "",
             f"{'metric':<26} {'micro':>8} {'macro':>8}",
             f"{'detection recall@.5':<26} {rec_mi:>8.3f} {rec_ma:>8.3f}",
             f"{'naming acc | found':<26} {name_mi:>8.3f} {name_ma:>8.3f}",
             f"{'END-TO-END (found&named)':<26} {e2e_mi:>8.3f} {e2e_ma:>8.3f}",
             "",
             "Phase-2 upper bound (GT boxes): micro 0.978  macro 0.891",
             "",
             f"{'id':>3} {'class':<28} {'gt':>5} {'rec':>6} {'name|f':>7} {'e2e':>6}",
             "-"*62]
    for c in sorted(present, key=lambda c: named_ok[c]/gt_n[c]):
        r = found[c]/gt_n[c]; nf = named_ok[c]/found[c] if found[c] else 0.0
        e = named_ok[c]/gt_n[c]
        lines.append(f"{c:>3} {names[c][:28]:<28} {gt_n[c]:>5} {r:>6.3f} {nf:>7.3f} {e:>6.3f}")
    report = "\n".join(lines)
    print("\n"+report)
    Path(args.out).write_text(report)
    print(f"\nwritten -> {args.out}\npredictions -> {args.pred_json}")


if __name__ == "__main__":
    main()
