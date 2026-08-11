"""Orchestrator: prepare -> train -> infer -> evaluate -> visualize.

A single multi-class YOLO detector both localizes and names objects — there is
no separate naming stage. Each step reads/writes artifacts under
`project.work_dir`, so steps can be run independently or chained with `run_all`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dataset as ds
from .config import Config
from .metrics import BoxEval, iou_matrix
from .phase1 import YoloProposer, train_detector
from .visualize import render_video


def _splits(cfg: Config):
    """Return (stems, annotated, remaining, eval_target, monitor).

    annotated   = frames we 'manually annotate' (used for training) — spread evenly/
                 randomly for a fixed budget (`sampling: even|random`), or picked by
                 greedy set cover for a per-class sample floor (`sampling: greedy`,
                 see `dataset.greedy_annotate_split`).
    remaining   = every other frame — what we auto-annotate.
    eval_target = `remaining`, optionally capped by data.eval_frames for speed.
    monitor     = small even subset of `remaining` for the detector's per-epoch val.
    """
    stems = ds.list_frames(cfg.images_dir, cfg.data.image_ext)
    sampling = getattr(cfg.data, "sampling", "even")
    seed = cfg.data.seed
    if sampling == "greedy":
        target = int(cfg.data.samples_per_class)
        annotated, remaining = ds.greedy_annotate_split(cfg.images_dir, cfg.data.image_ext,
                                                         stems, len(cfg.classes), target)
    else:
        annotated, remaining = ds.annotate_split(stems, int(cfg.data.annotation_frames), sampling, seed)
    ef = int(getattr(cfg.data, "eval_frames", -1) or -1)
    eval_target = ds.even_subset(remaining, ef, "even", seed) if ef > 0 else remaining
    mv = min(int(getattr(cfg.data, "monitor_val", 200)), len(remaining))
    monitor = ds.even_subset(remaining, mv, "even", seed)
    return stems, annotated, remaining, eval_target, monitor


# ---- steps -----------------------------------------------------------------

def prepare(cfg: Config) -> Path:
    _, annotated, remaining, eval_target, monitor = _splits(cfg)
    shuffle_train = bool(getattr(cfg.data, "shuffle_train_order", False))
    data_yaml = ds.build_yolo_dataset(cfg.images_dir, cfg.data.image_ext,
                                      annotated, monitor, cfg.dataset_dir,
                                      names=cfg.classes,
                                      shuffle_train_names=shuffle_train,
                                      seed=cfg.data.seed)
    print(f"[prepare] annotate {len(annotated)} frames (evenly spread) -> auto-annotate "
          f"{len(remaining)} remaining (scoring {len(eval_target)}); "
          f"detector monitor-val={len(monitor)} -> {data_yaml}"
          + (" [train filenames shuffled — same frames, different listing order]" if shuffle_train else ""))
    return data_yaml


def train(cfg: Config):
    data_yaml = cfg.dataset_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = prepare(cfg)
    best = train_detector(cfg, data_yaml)
    print(f"[train] best weights -> {best}")
    return best


def _read_gt(txt: Path, W: int, H: int):
    """(class_id, x1, y1, x2, y2) in pixels from a YOLO label file."""
    out = []
    for line in Path(txt).read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            c = int(float(p[0])); xc, yc, w, h = map(float, p[1:])
            out.append((c, (xc - w / 2) * W, (yc - h / 2) * H,
                        (xc + w / 2) * W, (yc + h / 2) * H))
    return out


def infer(cfg: Config, weights: Path | None = None) -> dict:
    _, _, _, eval_target, _ = _splits(cfg)
    w = Path(weights) if weights else cfg.best_weights
    if not w.exists():
        raise FileNotFoundError(f"detector weights not found: {w} (run `train` first)")
    proposer = YoloProposer(cfg, w)
    classes = cfg.classes
    images_dir, ext = cfg.images_dir, cfg.data.image_ext

    print(f"[infer] multi-class YOLO on {len(eval_target)} frames ...")
    preds = {}
    for i in range(0, len(eval_target), 64):
        batch = eval_target[i:i + 64]
        paths = [str(images_dir / f"{s}.{ext}") for s in batch]
        res = proposer.predict(paths)
        for s, r in zip(batch, res):
            H, W = r.orig_shape
            gt = _read_gt(images_dir / f"{s}.txt", W, H)
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

    cfg.predictions_json.write_text(json.dumps(preds), encoding="utf-8")
    print(f"[infer] predictions -> {cfg.predictions_json}")
    return preds


def _score_predictions(preds: dict, nC: int, thr):
    """Core class-aware + localization-only accumulation shared by `evaluate`.
    Pure numpy/Hungarian matching over an already-computed predictions.json."""
    be_ca, be_loc = BoxEval(thr), BoxEval(thr)
    gt_n = np.zeros(nC, int); found = np.zeros(nC, int); named = np.zeros(nC, int); fp = 0

    for d in preds.values():
        pred = d["pred"]
        pr_cls = [o["cls"] for o in pred]
        gt_ca = [(o["cls"], o["box"]) for o in d["gt"]]
        pr_ca = [(c, o["box"]) for c, o in zip(pr_cls, pred)]
        be_ca.add_frame(gt_ca, pr_ca)
        be_loc.add_frame([(0, o["box"]) for o in d["gt"]], [(0, o["box"]) for o in pred])

        gt = d["gt"]
        prb = np.array([o["box"] for o in pred], float) if pred else np.zeros((0, 4))
        prc = np.array(pr_cls, int)
        if gt:
            gb = np.array([o["box"] for o in gt], float)
            M = iou_matrix(gb, prb)
            matched = np.zeros(len(pred), bool)
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
    return be_ca, be_loc, gt_n, found, named, fp


def _frame_stats(cfg: Config, stems: list[str]):
    """Per-class box counts over exactly `stems` (not the whole dataset).
    Returns (per_class_counts, n_boxes)."""
    nC = len(cfg.classes)
    counts = np.zeros(nC, int)
    n_boxes = 0
    for s in stems:
        rows = ds._read_label(cfg.images_dir / f"{s}.txt")
        for r in rows:
            c = int(float(r[0]))
            if 0 <= c < nC:
                counts[c] += 1
        n_boxes += len(rows)
    return counts, n_boxes


def evaluate(cfg: Config) -> str:
    """Build the run report."""
    preds = json.loads(cfg.predictions_json.read_text())
    thr = tuple(cfg.eval.iou_thresholds)
    nC = len(cfg.classes)
    be_ca, be_loc, gt_n, found, named, fp = _score_predictions(preds, nC, thr)

    ca, loc = be_ca.summary(), be_loc.summary()
    present = [c for c in range(nC) if gt_n[c] > 0]
    def macro(num, den): return float(np.mean([num[c] / den[c] for c in present if den[c]]))
    rec_mi = found[present].sum() / gt_n[present].sum()
    name_mi = named[present].sum() / max(found[present].sum(), 1)
    e2e_mi = named[present].sum() / gt_n[present].sum()

    _, annotated, remaining, eval_target, _ = _splits(cfg)
    tr_counts, tr_n_boxes = _frame_stats(cfg, annotated)

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
        f"- **Detector:** multi-class YOLO  ·  confidence {cfg.infer.conf}",
        f"- **Predicted boxes:** {ca['num_pred']}  ·  **false positives:** {fp}  ·  **mean IoU:** {ca['mean_iou']}",
        "",
        "## Training frames (annotated set)",
        "",
        f"- **Frames:** {len(annotated)}  ·  **classes represented:** "
        f"{sum(1 for n in tr_counts if n > 0)}/{len(cfg.classes)}  ·  **GT boxes:** {tr_n_boxes}",
        "",
        "| id | class | gt boxes | % of total |",
        "|---:|---|---:|---:|",
    ]
    for c in sorted(range(len(cfg.classes)), key=lambda c: -tr_counts[c]):
        pct = 100 * tr_counts[c] / tr_n_boxes if tr_n_boxes else 0.0
        md.append(f"| {c} | {cfg.classes[c]} | {tr_counts[c]} | {pct:.1f}% |")
    md += [
        "",
        f"<details><summary>Frame list ({len(annotated)})</summary>",
        "",
        ", ".join(annotated),
        "",
        "</details>",
    ]
    md += [
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
    report_path.write_text(report, encoding="utf-8")
    print(f"[evaluate] -> {report_path}")
    return report


def visualize(cfg: Config):
    return render_video(cfg)


_STEP_FUNCS = {"prepare": prepare, "train": train, "infer": infer,
               "evaluate": evaluate, "visualize": visualize}


def run_all(cfg: Config):
    """Run the full pipeline in canonical order, skipping any step disabled via
    `project.steps` in the config (see `Config._load_steps`)."""
    for name, fn in _STEP_FUNCS.items():
        if name not in cfg.steps:
            print(f"[run_all] skip {name} (disabled in config)")
            continue
        fn(cfg)
