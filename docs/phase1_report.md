# Phase 1 — Localization (Region Proposal)

**Status: Phase-1 trained and validated** (YOLO11-L, 20 epochs). Localization recall is
≈99.9% — see [Results](#results). Next: wire Phase1→Phase2 end-to-end. This report
documents the setup, how to run/monitor the code, and the results. See
[plan.md](plan.md) §12 for the full approach menu; this implements **Approach A — a
fine-tuned single-class detector**.

---

## What Phase 1 does

The pipeline is two decoupled stages:

```
frame ─▶ [Phase 1: find objects → class-agnostic boxes] ─▶ crops ─▶ [Phase 2: DINOv2 kNN → class name]
```

Because Phase 2 already does the naming (validated at ~96% overall / ~89%
fair-average — see [phase2_report.md](phase2_report.md)), **Phase 1 only has to find
objects, not classify them.** That makes it a **class-agnostic, recall-first** problem:

- **Class-agnostic** — every GT box is collapsed to a single `object` class. Adding a
  new class later needs **no Phase-1 retraining** (its job is unchanged).
- **Recall-first** — a missed object is unrecoverable (no box → no reviewer sees it),
  whereas false boxes are cheap: Phase 2's rejection threshold drops proposals that
  match no class, and the rest go to review. So we tune Phase 1 to over-propose.

---

## Design choice: single-class vs. multi-class detector

Should the Phase-1 detector be trained on a **single `object` class** (naming left to
Phase 2), or on **all 27 classes** (detect + name in one model)? We chose single-class,
and the reasoning is central to the whole two-phase design.

**Finding an object and naming it are different-difficulty problems.** Localizing a
`hand_w_fixture` is easy — it's clearly a thing in the frame. *Naming* it vs.
`r_hand_w_allen_wrench` is hard (Phase 2 needs many exemplars and still only reaches
~89% fair-average). Merging classes into the detector forces it to solve the hard
problem just to do the easy one, and the rare classes pay for it.

| Factor | Single-class ("object") | Multi-class (27 classes) |
|---|---|---|
| **Rare-class recall** | Learns "objectness" from **all 78k boxes pooled** → rare classes localize fine | Each class head sees only its own boxes (e.g. `install_gaskit` 79, `r_hand_w_part` 214) → starved, boxes missed |
| **Class imbalance** | None — one class | Severe (`black_bin` 28k vs 79) → dominated by common classes, same long-tail problem as Phase 2 |
| **Recall behavior** | Box fires or not — no class ambiguity to suppress it | Ambiguous rare class → low confidence → **missed at threshold** |
| **Changing class list** | Proposer unchanged when a class is added/removed — **no retrain** | Add/remove a class → retrain the detector |
| **Separation of concerns** | Naming handed to Phase 2 (DINOv2 kNN), purpose-built for it | Detector re-does the hard part with less capacity |

**Empirical clincher:** the single-class detector reached **99.8% recall, ≥97% for
every class** — including the exact rare hand-poses Phase 2 struggles to *name*. A
27-class detector would almost certainly not match that rare-class recall: it inherits
Phase 2's long-tail difficulty **plus** the imbalance problem, and drops boxes whenever
it is unsure of the class.

**When multi-class is still worth having** — not as the proposer, but as a **second
source** (§3 / §12.3 dual-source voting): a multi-class YOLO gives a fast per-class
opinion on the **common** classes where it is strong, useful to cross-check Phase 2
(agreement → auto-accept, disagreement → review). It is also a clean baseline for
"does joint detect+classify beat decoupled Phase1→Phase2?" — expected to lose on the
rare classes, but a fair comparison.

> **Decision:** single-class detector as the localizer (this run). Optionally train a
> multi-class version later purely as a comparison / second voting source — not to
> replace the proposer.

---

## Setup

| Item | Value |
|---|---|
| Approach | A — fine-tuned single-class ("object") detector |
| Model | YOLO11-L (Ultralytics 8.4.46), local weights `yolov5/data/yolo11l.pt` |
| Dataset | `gas_valve_2view` — 7,748 frames, 1280×480 (two views side-by-side) |
| Labels | all 27 classes collapsed → class 0 `object` |
| Split | **same as Phase 2**: val = even-stride 1,000 held-out frames (11,424 boxes); train = other 6,748 frames (78,062 boxes) |
| Config | imgsz 1024, AMP on, batch 8 (auto from 16), 60 epochs, patience 20, close_mosaic 10, seed 0 |
| Compute | 1× RTX 4090 (GPU 0); ~3 min/epoch, ~2.5–3 h total |
| Env | conda `new_eda_tool` → `/home/retrocausal-train/anaconda3/envs/new_eda_tool/bin/python` |

> **Config notes:** imgsz 1280 without AMP hit a cuDNN workspace OOM; **imgsz 1024 with
> AMP** is the stable high-res config for these frames. Ultralytics' OOM guard reduced
> batch 16 → 8 automatically.

**Run directory:** `phase1/runs/yolo11l_obj-4/`
(the earlier `yolo11l_obj` / `-2` dirs are from two failed start attempts and can be
deleted).

---

## How to run the code

All commands from the project root
`/home/retrocausal-train/Documents/RPN+DinoV2`, using the env python
`PY=/home/retrocausal-train/anaconda3/envs/new_eda_tool/bin/python`.

### 1. Build the class-agnostic dataset
Collapses every box to `object`, reuses the Phase-2 split, symlinks images:
```bash
$PY scripts/phase1_build_dataset.py
# -> phase1/dataset/{images,labels}/{train,val} + data.yaml
```

### 2. Train
```bash
$PY scripts/phase1_train.py                 # YOLO11-L, imgsz 1024, 60 epochs, GPU 0
# swap model:  $PY scripts/phase1_train.py --rtdetr --model rtdetr-l.pt --name rtdetr_obj
# other flags: --imgsz --epochs --batch --device --data --no-amp
```

### 3. Evaluate recall
Recall broken out by the **original** class, on the 1,000-frame val split:
```bash
$PY scripts/phase1_eval.py \
  --weights phase1/runs/yolo11l_obj-4/weights/best.pt \
  --device 0
# -> phase1/recall_report.txt
# recall-first defaults: conf 0.001, iou_nms 0.7, imgsz 1024, max_det 300
```

---

## How to see training progress

**Clean per-epoch metrics** (best view; updates every epoch):
```bash
column -t -s, phase1/runs/yolo11l_obj-4/results.csv | cut -c1-90
```

**Live current epoch / iteration** (the moving progress line):
```bash
tr '\r' '\n' < phase1/train_yolo11l.log | grep -E '/60' | tail -1
# e.g. "5/60 ... 491/844 5.6it/s 1:32<1:03" = epoch 5 of 60, batch 491/844, ~1 min left this epoch
```

**Auto-refresh every 30 s** (Ctrl-C to stop):
```bash
watch -n 30 "tail -3 phase1/runs/yolo11l_obj-4/results.csv | column -t -s,"
```

**Visual curves** (open in IDE; refreshes each epoch):
```
phase1/runs/yolo11l_obj-4/results.png
```

**GPU utilization / is it still running:**
```bash
nvidia-smi
pgrep -af phase1_train.py
```

Columns of interest in `results.csv`: `metrics/precision(B)`, `metrics/recall(B)`,
`metrics/mAP50(B)`, `metrics/mAP50-95(B)`.

---

## Metrics we report (and how to read them)

Localization is scored by box overlap (IoU), independent of the name:

- **Recall @ IoU 0.5** — did each true object get a proposal box. The headline.
- **Micro vs. macro recall** — same lens as Phase 2: *micro* weights every box equally
  (dominated by common objects), *macro* weights every class equally (exposes weak
  rare/tiny classes). Watch the gap.
- **Average Recall (AR .5:.95)** — recall averaged over IoU 0.50→0.95; rewards tight
  boxes.
- **Per-class miss count** — how many boxes of each original class were not found.
- **Proposals / frame** — the cost Phase 2 + reviewers pay for recall.

Precision / mAP is secondary here (false boxes are filtered downstream), but the
training `results.csv` tracks it so we know the review-queue size.

---

## Results

> ⚠️ **These are full-data *ceiling* numbers, not deployment numbers.** The detector
> here was trained on **all 6,748** non-val frames (and the Phase-2 reference uses all
> their crops). The pipeline's real purpose is few-shot — annotate ~N frames,
> auto-label the rest — so this run only establishes the **upper bound**: it shows
> **localization is not the bottleneck** (≈99.8% recall when data is plentiful). The
> deployment-relevant question — *how few annotated frames still give good recall* — is
> the **annotation-budget sweep** below, which trains the detector (and builds the
> Phase-2 reference) from only N frames. See the pipeline config's
> `phase1.train_frames` / `phase2.reference_frames`.

**Training: stopped at epoch 20/60** (manually — recall and mAP50 had saturated; only
box-tightness/mAP50-95 was still creeping up, not worth the GPU time). Best checkpoint:
**epoch 20**, `phase1/runs/yolo11l_obj-4/weights/best.pt`.

### Training metrics (own-threshold, from `results.csv`)

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|--:|--:|--:|--:|--:|
| 1 | 0.935 | 0.917 | 0.962 | 0.755 |
| 5 | 0.970 | 0.953 | 0.980 | 0.832 |
| 10 | 0.980 | 0.963 | 0.982 | 0.860 |
| 15 | 0.982 | 0.968 | 0.989 | 0.879 |
| **20** | **0.982** | **0.971** | **0.989** | **0.886** |

Recall/mAP50 plateaued by ~epoch 5; the epochs after that bought **box tightness**
(mAP50-95 0.832 → 0.886).

### Phase-1 recall eval (`phase1_eval.py` on epoch-20 `best.pt`, 1,000 val frames)

Two operating points:

| Operating point | Proposals/frame | Micro recall@.5 | Macro recall@.5 | Micro AR .5:.95 | Macro AR .5:.95 |
|---|--:|--:|--:|--:|--:|
| **conf 0.001** (recall-first) | 27.5 | **99.9%** | **99.7%** | 0.894 | 0.792 |
| **conf 0.1** (review-load) | **12.2** | 99.4% | 97.8% | 0.889 | 0.773 |

- **Localization is effectively solved for *finding* objects.** Recall@.5 ≥ 97% for
  **every** class; only **~14 of 11,424 boxes** missed at conf 0.001. Micro ≈ macro
  recall — **no long-tail gap** (unlike Phase-2 naming).
- **The hard-to-*name* classes are easy to *find*:** `hand_w_fixture` 97.3%,
  `r_hand_w_allen_wrench` 97.6%, `install_gaskit_pose` 100%. Validates the find/name
  split.
- **conf 0.1 is a great operating point:** 12.2 proposals/frame ≈ the ~11.4 true
  objects/frame (almost no false boxes), at 99.4% micro / 97.8% macro recall. conf
  0.001 squeezes the last 0.1–2% recall at the cost of ~2× the proposals — worth it
  since Phase 2's rejection filters the extra.
- **AR .5:.95** (box tightness) is the only sub-saturated metric — 0.894 micro. Extra
  training lifted it (+4.5 pts vs epoch 5); RT-DETR or more epochs could push further,
  but boxes are already tight enough for review.

### Per-class recall @IoU 0.5 (conf 0.001, worst first)

| Class | GT boxes | rec@.5 | AR .5:.95 | missed |
|---|--:|--:|--:|--:|
| hand_w_fixture | 74 | 0.973 | 0.643 | 2 |
| r_hand_w_allen_wrench | 41 | 0.976 | 0.495 | 1 |
| f1_BR_hand_w_screwdriver | 44 | 0.977 | 0.789 | 1 |
| right_hand_w_screwdriver | 389 | 0.995 | 0.754 | 2 |
| left_hand_w_screwdriver | 374 | 0.997 | 0.828 | 1 |
| hands | 2448 | 0.998 | 0.828 | 4 |
| screw_bin | 1538 | 0.999 | 0.897 | 2 |
| parts_bin | 775 | 0.999 | 0.924 | 1 |
| *(all 18 other classes)* | — | **1.000** | 0.60–0.97 | 0 |

Full reports: [phase1/recall_report.txt](../phase1/recall_report.txt) (conf 0.001) and
`phase1/recall_report_conf0.1.txt`.

---

## End-to-end Phase 1 → Phase 2

The payoff test: run the detector (conf 0.1), crop each proposal, name it with the
DINOv2 kNN classifier (all-frames reference), match to GT (IoU ≥ 0.5), and score the
compounded box **and** name against the Phase-2 GT-box upper bound.

| Metric | Micro | Macro |
|---|--:|--:|
| Detection recall @.5 | 0.994 | 0.978 |
| Naming acc \| found | 0.970 | 0.833 |
| **End-to-end (found & named)** | **0.964** | **0.819** |
| *Phase-2 upper bound (GT boxes)* | *0.978* | *0.891* |

Predicted boxes: 12,193 (12.2/frame); false positives (no GT match): 1,020 (8.4%).

**What it says:**
- **The full auto-annotation pipeline works.** Micro end-to-end **96.4%** is within
  **1.4 points** of the 97.8% GT-box ceiling — i.e. localization costs almost nothing
  on the common classes; ~96% of *all* boxes are both found and correctly named
  automatically.
- **The macro gap is bigger (81.9% vs 89.1%, ~7 pts)** and it is a **naming** loss, not
  a detection loss: detection recall is 97.8% macro (rare objects *are* found), but
  `naming acc | found` drops to 83.3% macro. Predicted boxes are slightly looser than
  GT, and that box-tightness shift moves the crop embedding — which the already-hard
  fine-grained rare classes (`hand_w_fixture` 25%, `r_hand_w_allen_wrench` 51%, several
  `f2_*` poses) are most sensitive to (the crop-sensitivity failure mode, plan §6).
- **Takeaway:** the pipeline is production-ready for the common/distinct classes
  (auto-label + spot-check). The rare fine-grained hand-poses remain the weak spot, and
  the lever for them is **tighter boxes** (RT-DETR / more AR) plus Phase-2's confidence
  gate — not more detection.

**Standard detection metrics** (class-aware, [acp-agentic-workflow](../../acp-agentic-workflow/src/evaluation/box_metrics.py) convention, conf 0.1):
F1 **0.936** / accuracy 0.879 @IoU0.5, F1 **0.902** / accuracy 0.822 @IoU0.75. Full
P/R/F1/accuracy tables (end-to-end and localization-only) are in the dedicated
**[end-to-end report → e2e_report.md](e2e_report.md)**.

Report: [phase1/e2e_report.txt](../phase1/e2e_report.txt) · per-frame predictions:
`phase1/e2e_predictions.json`.

### Visualization

`phase1/e2e_pred_vs_gt.mp4` — stacked comparison video (1280×960, 500 val frames):
**top = predictions** (red boxes + predicted class tag), **bottom = ground truth**
(green boxes + class tag). Lets you eyeball naming mistakes and false positives frame
by frame.

```bash
$PY scripts/phase1_visualize.py --limit 500 --fps 8     # rebuild; --limit -1 for all 1000
```

### Still to do / optional

- [ ] RT-DETR-L run for tighter boxes (AR) — the main lever for the rare-class naming gap.
- [ ] Multi-class YOLO as a second voting source (see the single-vs-multi discussion above).
- [ ] SAHI / `yolo26-p2` if tiny objects (screws) become a focus.
- [ ] Use Phase-2 rejection to prune the 1,020 false positives before review.

---

## Files

| File | Purpose |
|---|---|
| [phase1_build_dataset.py](../scripts/phase1_build_dataset.py) | Build class-agnostic YOLO dataset (Phase-2 split) |
| [phase1_train.py](../scripts/phase1_train.py) | Train single-class detector (YOLO11-L / RT-DETR) |
| [phase1_eval.py](../scripts/phase1_eval.py) | Recall-first eval, per original class |
| [phase1_phase2_e2e.py](../scripts/phase1_phase2_e2e.py) | End-to-end: detect → DINOv2 kNN name → score + save predictions |
| [phase1_visualize.py](../scripts/phase1_visualize.py) | Stacked pred-vs-GT comparison video |
| `phase1/dataset/` | Symlinked images + single-class labels + `data.yaml` |
| `phase1/runs/yolo11l_obj-4/` | Training run: `weights/best.pt`, `results.csv`, `results.png` |
| `phase1/e2e_report.txt`, `e2e_predictions.json`, `e2e_pred_vs_gt.mp4` | End-to-end results, per-frame preds, video |
| `phase1/train_yolo11l.log` | Full training stdout (progress bars) |
