# End-to-End Evaluation — Phase 1 → Phase 2

Full auto-annotation pipeline scored on the held-out **1,000-frame** val set (the same
split as Phase 2, so nothing here was seen in training or in the naming reference).

> ⚠️ **Ceiling, not deployment numbers.** The detector was trained on **all 6,748**
> non-val frames and the kNN reference uses **all** their crops. That over-supplies
> both phases relative to the real few-shot goal (annotate ~N frames → auto-label the
> rest). Read these as the **upper bound**. The deployment-relevant numbers come from
> the **annotation-budget sweep** (train detector + build reference from only N frames)
> — run the pipeline with `phase1.train_frames = phase2.reference_frames = N`.

```
frame ─▶ [Phase 1: YOLO11-L single-class detector → boxes] ─▶ crops ─▶ [Phase 2: DINOv2 kNN → class name]
```

## Setup

| Item | Value |
|---|---|
| Detector | YOLO11-L single-class, `phase1/runs/yolo11l_obj-4/weights/best.pt` (epoch 20) |
| Namer | DINOv2-base + kNN (k=5) over the all-frames reference (78,060 GT crops) |
| Val set | 1,000 held-out frames — 11,424 GT boxes |
| Operating point | detector **conf 0.1**, NMS IoU 0.7 → 12,193 predicted boxes |
| Mean IoU (matched) | 0.923 |

## Detection metrics @ IoU 0.5 and 0.75

Scored with the `acp-agentic-workflow` convention
([src/evaluation/box_metrics.py](../../acp-agentic-workflow/src/evaluation/box_metrics.py)):
class-aware Hungarian matching per class; **precision** = TP/#pred, **recall** =
TP/#GT, **F1** = harmonic mean, **accuracy** = TP/(TP+FP+FN) (Jaccard / critical
success index — detection has no true negatives).

**End-to-end (class-aware — box *and* name must be correct):**

| IoU | Precision | Recall | F1 | Accuracy |
|---|--:|--:|--:|--:|
| **0.50** | 0.906 | 0.967 | **0.936** | 0.879 |
| **0.75** | 0.874 | 0.932 | **0.902** | 0.822 |

**Localization only (class-agnostic — box correct, name ignored):**

| IoU | Precision | Recall | F1 | Accuracy |
|---|--:|--:|--:|--:|
| **0.50** | 0.918 | 0.980 | **0.948** | 0.902 |
| **0.75** | 0.884 | 0.944 | **0.913** | 0.840 |

**Reading it:**
- **Naming costs almost nothing** — class-aware vs class-agnostic differ by ~1–1.5
  points (F1@0.5 0.936 vs 0.948). Nearly every localized box is also named correctly.
- **Box tightness is the IoU@0.75 penalty** — every metric drops ~4 points from 0.5 →
  0.75 despite mean IoU 0.923. This is the AR/tightness gap (the lever RT-DETR or more
  epochs would move).
- **Precision (~0.91) is held down by ~1,020 false positives** at conf 0.1 (unmatched
  predictions — duplicates or unannotated objects). Raising conf trades recall for
  precision and lifts accuracy; see the operating-point note.

## Compounded per-class view (naming lens)

Same run, matched class-agnostically then checking the name — separates "found" from
"named", and shows the long tail. (Recall differs slightly from the table above:
here a GT is "found" if **any** prediction overlaps ≥0.5; the table above uses strict
one-to-one Hungarian matching.)

| Metric | Micro (per box) | Macro (per class) |
|---|--:|--:|
| Detection recall @.5 | 0.994 | 0.978 |
| Naming acc \| found | 0.970 | 0.833 |
| **End-to-end (found & named)** | **0.964** | **0.819** |
| *Phase-2 upper bound (GT boxes)* | *0.978* | *0.891* |

The **macro** gap vs the upper bound (81.9% vs 89.1%) is a **naming** loss, not
detection: rare objects are found (97.8% macro recall) but predicted boxes are looser
than GT, and that shift degrades the fine-grained rare-class embeddings (crop
sensitivity, plan §6).

### Per original class — end-to-end (worst first)

| Class | GT | recall | name\|found | e2e |
|---|--:|--:|--:|--:|
| hand_w_fixture | 74 | 0.973 | 0.250 | 0.243 |
| f2_BR_hand_w_screwdriver | 41 | 1.000 | 0.415 | 0.415 |
| r_hand_w_allen_wrench | 41 | 0.854 | 0.514 | 0.439 |
| r_hand_w_complete_part | 18 | 0.833 | 0.533 | 0.444 |
| r_hand_w_part | 23 | 0.913 | 0.524 | 0.478 |
| f2_FR_hand_w_screwdriver | 58 | 1.000 | 0.483 | 0.483 |
| f1_FL_hand_w_screwdriver | 48 | 1.000 | 0.771 | 0.771 |
| f2_FL_hand_w_screwdriver | 40 | 1.000 | 0.775 | 0.775 |
| right_hand_w_screwdriver | 389 | 0.982 | 0.866 | 0.851 |
| left_hand_w_screwdriver | 374 | 0.981 | 0.896 | 0.880 |
| completed_part_on_white_bin | 62 | 1.000 | 0.903 | 0.903 |
| fixture_3 | 46 | 0.957 | 0.955 | 0.913 |
| f1_BR_hand_w_screwdriver | 44 | 0.955 | 0.976 | 0.932 |
| f1_FR_hand_w_screwdriver | 41 | 1.000 | 0.951 | 0.951 |
| f1_BL_hand_w_screwdriver | 45 | 1.000 | 0.956 | 0.956 |
| right_part_on_white_bin | 300 | 1.000 | 0.973 | 0.973 |
| hands | 2448 | 0.991 | 0.982 | 0.973 |
| rotated_part_on_white_bin | 352 | 1.000 | 0.974 | 0.974 |
| empty_white_box | 72 | 1.000 | 0.986 | 0.986 |
| screw_bin | 1538 | 0.995 | 0.992 | 0.988 |
| r_fixture1 | 178 | 0.994 | 0.994 | 0.989 |
| r_fixture2 | 231 | 0.996 | 0.996 | 0.991 |
| parts_bin | 775 | 0.996 | 0.999 | 0.995 |
| black_bin | 4134 | 1.000 | 0.999 | 0.999 |
| f2_BL_hand_w_screwdriver | 43 | 1.000 | 1.000 | 1.000 |
| install_gaskit_pose | 9 | 1.000 | 1.000 | 1.000 |

## Operating point note

All numbers are at detector **conf 0.1** (recall-favoring). Because precision and
accuracy are sensitive to the ~1,020 false positives, the operating point matters:
raising conf (0.25–0.5) raises precision/accuracy and trims recall slightly. Phase-2's
rejection threshold can also prune false proposals before review. A conf sweep is the
recommended next step to fix the deployment operating point.

## Visualization

`phase1/e2e_pred_vs_gt.mp4` — stacked comparison, 1,000 val frames @ 8 fps
(1280×960): **top = predictions** (red boxes + predicted class tag), **bottom = ground
truth** (green boxes + class tag).

## Reproduce

```bash
PY=/home/retrocausal-train/anaconda3/envs/new_eda_tool/bin/python
# detect -> name -> score + save per-frame predictions:
$PY scripts/phase1_phase2_e2e.py --device 0            # -> phase1/e2e_report.txt, e2e_predictions.json
# P/R/F1/accuracy @IoU0.5/0.75 (acp-agentic-workflow BoxEvalAccumulator convention):
$PY scripts/phase1_e2e_metrics.py                      # reads e2e_predictions.json
# comparison video:
$PY scripts/phase1_visualize.py --limit -1 --fps 8     # -> phase1/e2e_pred_vs_gt.mp4
```

## Files

| File | Purpose |
|---|---|
| [phase1_phase2_e2e.py](../scripts/phase1_phase2_e2e.py) | Detect → DINOv2 kNN name → score + save predictions |
| [phase1_e2e_metrics.py](../scripts/phase1_e2e_metrics.py) | P/R/F1/accuracy @IoU0.5/0.75 (acp-agentic-workflow convention) |
| [phase1_visualize.py](../scripts/phase1_visualize.py) | Stacked pred-vs-GT video |
| `phase1/e2e_report.txt` | Raw end-to-end + per-class output |
| `phase1/e2e_predictions.json` | Per-frame predictions (boxes, names, confidences) |
| `phase1/e2e_pred_vs_gt.mp4` | Comparison video |
