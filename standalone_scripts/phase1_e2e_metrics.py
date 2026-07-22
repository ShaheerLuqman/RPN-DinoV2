#!/usr/bin/env python3
"""Compute precision/recall/F1/accuracy @IoU0.5 and 0.75 for our Phase1->Phase2
predictions, using the acp-agentic-workflow BoxEvalAccumulator (identical convention).

Two views:
  end-to-end (class-aware) : pred class = DINOv2-kNN name; box AND name must match
  localization (class-agnostic): all classes -> 0; measures Phase-1 boxes only
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/retrocausal-train/Documents/acp-agentic-workflow")
from src.evaluation.box_metrics import BoxEvalAccumulator  # their exact metric

PRED = "/home/retrocausal-train/Documents/RPN+DinoV2/phase1/e2e_predictions.json"
data = json.loads(Path(PRED).read_text())

def run(class_aware):
    acc = BoxEvalAccumulator()
    for s, d in data.items():
        gt = [(o["cls"] if class_aware else 0, o["box"]) for o in d["gt"]]
        pred = [((o["cls"] if o["cls"] >= 0 else -99) if class_aware else 0, o["box"])
                for o in d["pred"]]
        acc.add_frame(gt, pred)
    return acc.summary()

for label, ca in [("END-TO-END (class-aware: box AND name)", True),
                  ("LOCALIZATION (class-agnostic: box only)", False)]:
    s = run(ca)
    print(f"\n=== {label} ===")
    print(f"  num_gt={s['num_gt']}  num_pred={s['num_pred']}  mean_iou={s['mean_iou']}")
    for thr in ("0.5", "0.75"):
        print(f"  IoU@{thr}:  precision={s['precision@'+thr]}  recall={s['recall@'+thr]}"
              f"  f1={s['f1@'+thr]}  accuracy={s['accuracy@'+thr]}")
