#!/usr/bin/env python3
"""Compare two Phase-1 multiclass detector checkpoints on the same dataset/config.

Runs `infer` + `evaluate` (the same steps `pipeline.cli` uses to produce a run
report) for each given checkpoint against a pipeline YAML config, then writes a
side-by-side comparison of the headline numbers.

Speed note: `phase2.fuse_multiclass` in the config makes `infer` also DINOv2-embed
every predicted crop, purely to build the bonus YOLO/DINOv2 weight-sweep table —
it does NOT affect any of the core Localization / With-classes / Compounded /
Per-class numbers (those are always the detector's own, pure-YOLO predictions).
This script disables fusion by default (`--fuse` to turn it back on) so a
detector-vs-detector comparison doesn't pay for ~700k DINOv2 embeddings per model.

A checkpoint whose weights are byte-identical to a previously-cached
`predictions_<label>.json` in the run's work_dir is not re-run (pass --force to
override) — e.g. if a report for these exact weights already exists from a normal
pipeline run.

Usage:
    python standalone_scripts/compare_detectors.py \\
        --config configs/doosan_swivel.yaml \\
        --model trained=compare_model/doosan_trained_detector.pt \\
        --model deployed=compare_model/doosan_deployed_detector.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_model_arg(s: str) -> tuple[str, Path]:
    label, sep, path = s.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"expected label=path/to/weights.pt, got {s!r}")
    return label, Path(path)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _headline(cfg, P, np, preds: dict) -> dict:
    """Aggregate metrics for the comparison table, computed the same way `evaluate`
    computes the numbers in its main report body (pure detector predictions,
    weight=None -> the class baked in at infer time)."""
    thr = tuple(cfg.eval.iou_thresholds)
    nC = len(cfg.classes)
    be_ca, be_loc, gt_n, found, named, fp = P._score_predictions(preds, nC, thr, weight=None)
    ca, loc = be_ca.summary(), be_loc.summary()
    present = [c for c in range(nC) if gt_n[c] > 0]

    def macro(num, den):
        return float(np.mean([num[c] / den[c] for c in present if den[c]]))

    rec_mi = found[present].sum() / gt_n[present].sum()
    name_mi = named[present].sum() / max(found[present].sum(), 1)
    e2e_mi = named[present].sum() / gt_n[present].sum()

    out = {
        "predicted_boxes": ca["num_pred"], "false_positives": fp, "mean_iou": ca["mean_iou"],
        "det_recall_micro": rec_mi, "det_recall_macro": macro(found, gt_n),
        "naming_acc_micro": name_mi, "naming_acc_macro": macro(named, found),
        "e2e_micro": e2e_mi, "e2e_macro": macro(named, gt_n),
    }
    for t in thr:
        k = f"{t:g}"
        for section, summ in (("loc", loc), ("cls", ca)):
            for metric in ("precision", "recall", "f1", "accuracy"):
                out[f"{section}_{metric}@{k}"] = summ[f"{metric}@{k}"]
    return out


def _comparison_md(cfg, labels: list[str], headlines: dict[str, dict]) -> str:
    thr = tuple(cfg.eval.iou_thresholds)
    lines = [f"# Detector comparison — {cfg.work.name}", "",
             "| model | " + " | ".join(labels) + " |",
             "|---" * (len(labels) + 1) + "|"]

    def row(name, fmt, *keys):
        vals = [fmt.format(headlines[l][k]) for l, k in zip(labels, keys)]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    lines += ["", "## Localization — box only (class-agnostic)", ""]
    for t in thr:
        k = f"{t:g}"
        lines.append(f"**IoU {k}**")
        lines.append("| model | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for metric, name in (("precision", "Precision"), ("recall", "Recall"),
                             ("f1", "F1"), ("accuracy", "Accuracy")):
            vals = [f"{headlines[l][f'loc_{metric}@{k}']:.3f}" for l in labels]
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")

    lines += ["## With classes — end-to-end (box + name)", ""]
    for t in thr:
        k = f"{t:g}"
        lines.append(f"**IoU {k}**")
        lines.append("| model | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for metric, name in (("precision", "Precision"), ("recall", "Recall"),
                             ("f1", "F1"), ("accuracy", "Accuracy")):
            vals = [f"{headlines[l][f'cls_{metric}@{k}']:.3f}" for l in labels]
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")

    lines += ["## Compounded (naming lens)", "",
             "| metric | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for key, name in (("det_recall_micro", "Detection recall @.5 (micro)"),
                      ("det_recall_macro", "Detection recall @.5 (macro)"),
                      ("naming_acc_micro", "Naming acc | found (micro)"),
                      ("naming_acc_macro", "Naming acc | found (macro)"),
                      ("e2e_micro", "End-to-end (micro)"),
                      ("e2e_macro", "End-to-end (macro)")):
        vals = [f"{headlines[l][key]:.3f}" for l in labels]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    lines += ["", "| metric | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|",
             "| Predicted boxes | " + " | ".join(str(headlines[l]["predicted_boxes"]) for l in labels) + " |",
             "| False positives | " + " | ".join(str(headlines[l]["false_positives"]) for l in labels) + " |",
             "| Mean IoU (matched) | " + " | ".join(f"{headlines[l]['mean_iou']:.4f}" for l in labels) + " |"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/doosan_swivel.yaml")
    ap.add_argument("--model", action="append", type=_parse_model_arg, required=True,
                    help="label=path/to/weights.pt, repeatable")
    ap.add_argument("--fuse", action="store_true",
                    help="honor phase2.fuse_multiclass (slow: DINOv2-embeds every "
                         "predicted crop for the bonus weight-sweep table). Off by default.")
    ap.add_argument("--force", action="store_true",
                    help="re-run infer even if a cached predictions_<label>.json matches these weights")
    args = ap.parse_args()

    from pipeline.config import Config
    cfg = Config(args.config)

    dev = str(cfg.phase1.device)
    if dev.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
        cfg.phase1.device = "0"
    if getattr(cfg.phase2, "hf_offline", False):
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            os.environ.setdefault(k, "1")
    if not args.fuse:
        cfg.phase2.fuse_multiclass = False

    import numpy as np
    from pipeline import pipeline as P

    labels, headlines = [], {}
    for label, weights in args.model:
        weights = weights if weights.is_absolute() else (REPO_ROOT / weights)
        if not weights.exists():
            raise FileNotFoundError(weights)
        labels.append(label)

        pred_out = cfg.work / f"predictions_{label}.json"
        hash_out = cfg.work / f"predictions_{label}.md5"
        report_out = cfg.work / f"{cfg.work.name}_report_{label}.md"
        wmd5 = _md5(weights)

        reuse = (not args.force and pred_out.exists() and hash_out.exists()
                and hash_out.read_text().strip() == wmd5)
        if reuse:
            print(f"[{label}] weights unchanged (md5 {wmd5[:12]}) — reusing cached {pred_out.name}")
            shutil.copy2(pred_out, cfg.predictions_json)
        else:
            print(f"[{label}] infer with {weights.name} (md5 {wmd5[:12]}) ...")
            P.infer(cfg, weights=weights)
            shutil.copy2(cfg.predictions_json, pred_out)
            hash_out.write_text(wmd5)

        print(f"[{label}] evaluate ...")
        report = P.evaluate(cfg)
        report_out.write_text(report, encoding="utf-8")
        preds = json.loads(cfg.predictions_json.read_text())
        headlines[label] = _headline(cfg, P, np, preds)
        print(f"[{label}] report -> {report_out}\n")

    comparison = _comparison_md(cfg, labels, headlines)
    comp_path = cfg.work / f"comparison_{'_vs_'.join(labels)}.md"
    comp_path.write_text(comparison, encoding="utf-8")
    print(comparison)
    print(f"\n[compare] -> {comp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
