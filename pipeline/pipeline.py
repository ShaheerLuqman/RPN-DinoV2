"""Orchestrator: prepare -> train -> reference -> infer -> evaluate -> visualize.

Each step reads/writes artifacts under `project.work_dir`, so steps can be run
independently or chained with `run_all`.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import dataset as ds
from .config import Config
from .metrics import BoxEval, iou_matrix
from .phase1 import (build_proposer, build_visual_exemplars, is_frcnn, is_multiclass,
                     is_training_free, proposer_kind, train_detector)
from ultralytics import YOLO, RTDETR
from .phase2 import Embedder, build_reference, classify_knn, crop_boxes, read_gt
from .visualize import render_video


def _splits(cfg: Config):
    """Return (stems, annotated, remaining, eval_target, monitor).

    annotated   = the N evenly-spread frames we 'manually annotate' (train + reference).
    remaining   = every other frame — what we auto-annotate.
    eval_target = `remaining`, optionally capped by data.eval_frames for speed.
    monitor     = small even subset of `remaining` for the detector's per-epoch val.
    """
    stems = ds.list_frames(cfg.images_dir, cfg.data.image_ext)
    sampling = getattr(cfg.data, "sampling", "even")
    seed = cfg.data.seed
    annotated, remaining = ds.annotate_split(stems, int(cfg.data.annotation_frames), sampling, seed)
    ef = int(getattr(cfg.data, "eval_frames", -1) or -1)
    eval_target = ds.even_subset(remaining, ef, "even", seed) if ef > 0 else remaining
    mv = min(int(getattr(cfg.data, "monitor_val", 200)), len(remaining))
    monitor = ds.even_subset(remaining, mv, "even", seed)
    return stems, annotated, remaining, eval_target, monitor


# ---- steps -----------------------------------------------------------------

def _check_label_ids(cfg: Config, yr) -> None:
    """Fail fast if a label class id falls outside classes.txt — the usual cause is
    deleting a line from classes.txt without remapping the (positional) label ids."""
    if len(yr) and int(yr.max()) >= len(cfg.classes):
        raise ValueError(
            f"label class id {int(yr.max())} is out of range for the "
            f"{len(cfg.classes)} classes in {cfg.resolve(cfg.data.classes_file)}.\n"
            f"Label files index classes.txt by line number, so do not remove a class "
            f"line without shifting every label id down. Restore the missing line "
            f"(a class with 0 boxes is harmless) or remap the labels.")


def prepare(cfg: Config) -> Path:
    _, annotated, remaining, eval_target, monitor = _splits(cfg)
    data_yaml = ds.build_yolo_dataset(cfg.images_dir, cfg.data.image_ext,
                                      annotated, monitor, cfg.dataset_dir,
                                      single_cls=not is_multiclass(cfg), names=cfg.classes)
    print(f"[prepare] annotate {len(annotated)} frames (evenly spread) -> auto-annotate "
          f"{len(remaining)} remaining (scoring {len(eval_target)}); "
          f"detector monitor-val={len(monitor)} -> {data_yaml}")
    return data_yaml


def train(cfg: Config):
    if is_training_free(cfg):
        print(f"[train] proposer '{proposer_kind(cfg)}' is training-free — skipping.")
        return None
    if is_frcnn(cfg):
        from .frcnn import train_frcnn
        _, annotated, *_ = _splits(cfg)
        return train_frcnn(cfg, annotated)
    data_yaml = cfg.dataset_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = prepare(cfg)
    best = train_detector(cfg, data_yaml)
    print(f"[train] best weights -> {best}")
    return best


def reference(cfg: Config, rebuild: bool = False):
    _, annotated, *_ = _splits(cfg)
    emb = Embedder(cfg.phase2.embed_model, cfg.phase1.device)
    Xr, yr = build_reference(cfg, emb, annotated, rebuild=rebuild)
    _check_label_ids(cfg, yr)
    print(f"[reference] {len(yr)} crops from {len(annotated)} annotated frames -> {cfg.reference_cache}")
    return Xr, yr


def _infer_multiclass(cfg: Config, eval_target, weights) -> dict:
    """Multi-class detector names boxes itself — Phase 2 skipped."""
    if proposer_kind(cfg) not in ("yolo11l", "yolo11x"):
        raise ValueError(f"phase1.multiclass needs a YOLO proposer "
                         f"(yolo11l/yolo11x), not '{proposer_kind(cfg)}'")
    w = Path(weights) if weights else cfg.best_weights
    if not w.exists():
        raise FileNotFoundError(f"detector weights not found: {w} (run `train` first)")
    model = (RTDETR if cfg.phase1.rtdetr else YOLO)(str(w))
    classes = cfg.classes
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    print(f"[infer] multi-class detector (Phase 2 skipped) on {len(eval_target)} frames ...")
    preds = {}
    for i in range(0, len(eval_target), 64):
        batch = eval_target[i:i + 64]
        paths = [str(images_dir / f"{s}.{ext}") for s in batch]
        res = model.predict(paths, conf=cfg.infer.conf, iou=cfg.infer.iou_nms,
                            imgsz=cfg.phase1.imgsz, max_det=cfg.infer.max_det,
                            device=str(cfg.phase1.device), verbose=False)
        for s, r in zip(batch, res):
            H, W = r.orig_shape
            gt = read_gt(images_dir / f"{s}.txt", W, H)
            pl = []
            if len(r.boxes):
                xy = r.boxes.xyxy.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for bx, cc, ff in zip(xy, cl, cf):
                    pl.append({"box": [round(float(bx[0]), 1), round(float(bx[1]), 1),
                                       round(float(bx[2]), 1), round(float(bx[3]), 1)],
                               "cls": int(cc), "name": classes[int(cc)],
                               "det_conf": round(float(ff), 3), "name_conf": round(float(ff), 3)})
            preds[s] = {"W": W, "H": H, "pred": pl,
                        "gt": [{"box": [round(g[1], 1), round(g[2], 1), round(g[3], 1), round(g[4], 1)],
                                "cls": g[0], "name": classes[g[0]]} for g in gt]}
    cfg.predictions_json.write_text(json.dumps(preds))
    print(f"[infer] predictions -> {cfg.predictions_json}")
    return preds


def infer(cfg: Config, weights: Path | None = None) -> dict:
    _, annotated, _, eval_target, _ = _splits(cfg)
    if is_multiclass(cfg):
        return _infer_multiclass(cfg, eval_target, weights)
    if is_training_free(cfg):
        if proposer_kind(cfg) in ("yoloe-visual", "yoloe_visual"):
            n = int(getattr(cfg.phase1, "yoloe_refer_frames", 3))
            ex = build_visual_exemplars(cfg, annotated, n)
            print(f"[infer] yoloe-visual: {len(ex)} exemplar reference frames")
            proposer = build_proposer(cfg, exemplars=ex)
        else:
            proposer = build_proposer(cfg)             # no checkpoint needed
    elif is_frcnn(cfg):
        ck = cfg.work / "frcnn.pt"
        if not ck.exists():
            raise FileNotFoundError(f"frcnn checkpoint not found: {ck} (run `train` first)")
        proposer = build_proposer(cfg, ck)
    else:
        w = Path(weights) if weights else cfg.best_weights
        if not w.exists():
            raise FileNotFoundError(f"detector weights not found: {w} (run `train` first)")
        proposer = build_proposer(cfg, w)
    emb = Embedder(cfg.phase2.embed_model, cfg.phase1.device)
    Xr, yr = build_reference(cfg, emb, annotated)
    _check_label_ids(cfg, yr)
    images_dir, ext = cfg.images_dir, cfg.data.image_ext

    crops, frame_boxes = [], {}
    print(f"[infer] proposer='{proposer_kind(cfg)}' on {len(eval_target)} frames ...")
    for i in range(0, len(eval_target), 64):
        batch = eval_target[i:i + 64]
        paths = [str(images_dir / f"{s}.{ext}") for s in batch]
        for s, (boxes, confs, img, _shape) in zip(batch, proposer.propose(paths)):
            cc, keep = crop_boxes(img, boxes, cfg.phase2.min_box_px)
            base = len(crops)
            crops += cc
            pos = {bi: base + ci for ci, bi in enumerate(keep)}
            frame_boxes[s] = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                               float(cf), pos.get(bi, -1)]
                              for bi, (b, cf) in enumerate(zip(boxes, confs))]

    print(f"[infer] naming {len(crops)} crops ...")
    Xq = emb.embed(crops, cfg.phase2.batch)
    pcls, pconf, psim = classify_knn(Xq, Xr, yr, cfg.phase2.knn_k)

    # Rejection gate: drop proposals whose nearest reference crop is too dissimilar
    # (they resemble no known object). 0 = keep everything.
    reject = float(getattr(cfg.phase2, "reject_below", 0.0) or 0.0)

    def kept(ci):
        return ci >= 0 and (reject <= 0 or psim[ci] >= reject)

    classes = cfg.classes
    preds = {}
    n_total = n_kept = 0
    for s in eval_target:
        im = cv2.imread(str(images_dir / f"{s}.{ext}"))
        H, W = im.shape[:2]
        gt = read_gt(images_dir / f"{s}.txt", W, H)
        fb = frame_boxes[s]
        n_total += len(fb)
        pred_list = []
        for b in fb:
            ci = b[5]
            if not kept(ci):
                continue
            pred_list.append({
                "box": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)],
                "cls": int(pcls[ci]), "name": classes[int(pcls[ci])],
                "det_conf": round(b[4], 3), "name_conf": round(float(pconf[ci]), 3),
                "name_sim": round(float(psim[ci]), 3)})
        n_kept += len(pred_list)
        preds[s] = {
            "W": W, "H": H, "pred": pred_list,
            "gt": [{"box": [round(g[1], 1), round(g[2], 1), round(g[3], 1), round(g[4], 1)],
                    "cls": g[0], "name": classes[g[0]]} for g in gt],
        }
    cfg.predictions_json.write_text(json.dumps(preds))
    if reject > 0:
        print(f"[infer] reject_below={reject}: kept {n_kept}/{n_total} proposals "
              f"({n_total - n_kept} rejected)")
    print(f"[infer] predictions -> {cfg.predictions_json}")
    return preds


def evaluate(cfg: Config) -> str:
    preds = json.loads(cfg.predictions_json.read_text())
    thr = tuple(cfg.eval.iou_thresholds)
    be_ca, be_loc = BoxEval(thr), BoxEval(thr)
    nC = len(cfg.classes)
    gt_n = np.zeros(nC, int); found = np.zeros(nC, int); named = np.zeros(nC, int); fp = 0

    for d in preds.values():
        gt_ca = [(o["cls"], o["box"]) for o in d["gt"]]
        pr_ca = [((o["cls"] if o["cls"] >= 0 else -99), o["box"]) for o in d["pred"]]
        be_ca.add_frame(gt_ca, pr_ca)
        be_loc.add_frame([(0, o["box"]) for o in d["gt"]], [(0, o["box"]) for o in d["pred"]])

        gt = d["gt"]
        prb = np.array([o["box"] for o in d["pred"]], float) if d["pred"] else np.zeros((0, 4))
        prc = np.array([o["cls"] for o in d["pred"]])
        if gt:
            gb = np.array([o["box"] for o in gt], float)
            M = iou_matrix(gb, prb)
            matched = np.zeros(len(d["pred"]), bool)
            for gi, o in enumerate(gt):
                c = o["cls"]; gt_n[c] += 1
                if prb.shape[0] == 0:
                    continue
                j = int(M[gi].argmax())
                if M[gi, j] >= 0.5:
                    found[c] += 1; matched[j] = True
                    if prc[j] == c:
                        named[c] += 1
            fp += int((~matched).sum())

    ca, loc = be_ca.summary(), be_loc.summary()
    present = [c for c in range(nC) if gt_n[c] > 0]
    def macro(num, den): return float(np.mean([num[c] / den[c] for c in present if den[c]]))
    rec_mi = found[present].sum() / gt_n[present].sum()
    name_mi = named[present].sum() / max(found[present].sum(), 1)
    e2e_mi = named[present].sum() / gt_n[present].sum()

    _, annotated, remaining, eval_target, _ = _splits(cfg)
    ref_n = int(np.load(cfg.reference_cache)["yr"].shape[0]) if cfg.reference_cache.exists() else 0

    def rows(summary):
        out = []
        for t in thr:
            k = f"{t:g}"
            out.append(f"| {k} | {summary['precision@'+k]:.3f} | {summary['recall@'+k]:.3f} "
                       f"| {summary['f1@'+k]:.3f} | {summary['accuracy@'+k]:.3f} |")
        return "\n".join(out)

    md = [
        f"# Run report — {cfg.work.name}",
        "",
        f"- **Annotated frames (input):** {len(annotated)} — spread evenly across the dataset",
        f"- **Auto-annotated (remaining):** {len(remaining)}  ·  **scored:** {len(eval_target)}",
        f"- **Proposer:** `{proposer_kind(cfg)}`  ·  confidence {cfg.infer.conf}",
        (f"- **Namer:** detector itself (multi-class; Phase 2 skipped)" if is_multiclass(cfg)
         else f"- **Namer:** DINOv2 kNN (k={cfg.phase2.knn_k}) · reference = {ref_n} crops from the annotated frames"
         + (f" · reject_below={cfg.phase2.reject_below}" if float(getattr(cfg.phase2, 'reject_below', 0) or 0) > 0 else "")),
        f"- **Predicted boxes:** {ca['num_pred']}  ·  **false positives:** {fp}  ·  **mean IoU:** {ca['mean_iou']}",
        "",
        "## Localization — without classes (box only)",
        "",
        "How well objects are *found*, ignoring the predicted name.",
        "",
        "| IoU | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|",
        rows(loc),
        "",
        "## With classes — end-to-end (box **and** correct name)",
        "",
        "A prediction counts only if it overlaps a GT box *and* the name matches.",
        "",
        "| IoU | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|",
        rows(ca),
        "",
        "*accuracy = TP / (TP + FP + FN) (Jaccard); no true negatives in detection.*",
        "",
        "## Compounded (naming lens)",
        "",
        "| Metric | Micro (per box) | Macro (per class) |",
        "|---|---:|---:|",
        f"| Detection recall @.5 | {rec_mi:.3f} | {macro(found, gt_n):.3f} |",
        f"| Naming acc \\| found | {name_mi:.3f} | {macro(named, found):.3f} |",
        f"| **End-to-end (found & named)** | **{e2e_mi:.3f}** | **{macro(named, gt_n):.3f}** |",
        "",
        "## Per-class end-to-end (worst first)",
        "",
        "| id | class | gt | recall | name\\|found | e2e |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for c in sorted(present, key=lambda c: named[c] / gt_n[c]):
        r = found[c] / gt_n[c]; nf = named[c] / found[c] if found[c] else 0.0
        md.append(f"| {c} | {cfg.classes[c]} | {gt_n[c]} | {r:.3f} | {nf:.3f} | {named[c]/gt_n[c]:.3f} |")
    md += ["", f"*Visualization: `{cfg.work.name}_viz.mp4` — top = predictions (red), bottom = ground truth (green).*"]

    report = "\n".join(md)
    report_path = cfg.work / f"{cfg.work.name}_report.md"
    report_path.write_text(report)
    print(f"[evaluate] -> {report_path}")
    return report


def visualize(cfg: Config):
    return render_video(cfg)


def run_all(cfg: Config):
    if not is_training_free(cfg):       # SAM/FastSAM/YOLOE are training-free
        if not is_frcnn(cfg):           # frcnn reads labels directly, no YOLO dataset
            prepare(cfg)
        train(cfg)
    infer(cfg)
    evaluate(cfg)
    visualize(cfg)
