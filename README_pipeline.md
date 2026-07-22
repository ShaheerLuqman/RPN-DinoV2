# Semi-Automated Annotation Pipeline

Two decoupled stages that turn a few annotated frames into auto-labels for the rest:

```
frame ─▶ [Phase 1: single-class detector → boxes] ─▶ crops ─▶ [Phase 2: DINOv2 kNN → class name]
```

- **Phase 1 (localize)** — a class-agnostic proposer finds *where* objects are. Pick one
  via `phase1.proposer`: a **fine-tuned YOLO** (`yolo11l`/`yolo11x`, boxes collapsed to one
  `object` class) or a **training-free** proposer — **SAM/FastSAM** (segment-everything) or
  **YOLOE** (open-vocab, prompt-free or visual-prompt with your annotated frames as
  exemplars). Training-free proposers skip the train step entirely.
- **Phase 2 (name)** — each detected crop is embedded with DINOv2 and labelled by a
  kNN vote against a reference library built from the annotated frames' crops. Adding a
  class needs no retraining — just more reference crops.

Config-driven and dataset-agnostic: point the YAML at any dataset of images + YOLO
labels and run.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt                        # install torch matching your CUDA
```

## Dataset layout

A flat directory of images, each with a sibling YOLO label file, plus `classes.txt`:

```
my_dataset/
  000001.png   000001.txt      # "class_id xc yc w h" (normalized), one box per line
  000002.png   000002.txt
  ...
  classes.txt                  # one class name per line; line index = class id
```

## Configure

Copy `configs/gas_valve.yaml`, edit the `data:` paths and (optionally) `project.root`
(`auto` = the repo dir containing `configs/`). Key knobs:

| Field | Meaning |
|---|---|
| `data.annotation_frames` | **the annotation budget** — how many frames you annotate (spread evenly). Trains the detector **and** builds the naming reference; every other frame is auto-annotated & scored. |
| `data.eval_frames` | how many of the remaining frames to score (`-1` = all remaining) |
| `data.monitor_val` | small held-out subset for the detector's per-epoch validation |
| `phase1.proposer` | **which localizer**: `yolo11l`/`yolo11x` (fine-tuned, trained) · `sam`/`fastsam` (segment-everything, training-free) · `yoloe` (open-vocab prompt-free) · `yoloe-visual` (open-vocab, annotated frames as exemplars) |
| `phase2.reject_below` | drop proposals whose nearest-reference cosine sim < this (filters SAM/YOLOE junk; 0 = keep all) |
| `infer.conf` | detector confidence (lower = more recall, more review load) |
| `eval.iou_thresholds` | IoU thresholds for P/R/F1/accuracy |

## Run

```bash
CFG=configs/gas_valve.yaml
python -m pipeline.cli prepare    --config $CFG   # build class-agnostic YOLO dataset
python -m pipeline.cli train      --config $CFG   # train the detector
python -m pipeline.cli reference  --config $CFG   # build the DINOv2 kNN reference
python -m pipeline.cli infer      --config $CFG   # detect + name the val frames
python -m pipeline.cli evaluate   --config $CFG   # P/R/F1/accuracy @ IoU + per-class
python -m pipeline.cli visualize  --config $CFG   # stacked pred-vs-GT video
# or everything:
python -m pipeline.cli all        --config $CFG
```

Artifacts land under `project.work_dir`:

```
runs/<name>/
  dataset/                  # symlinked images + single-class labels + data.yaml
  detector/train/weights/   # best.pt, last.pt
  reference.npz             # cached DINOv2 embeddings (Xr, yr)
  predictions.json          # per-frame boxes + names + confidences
  report.md                 # metrics (localization w/o classes + with classes + per-class)
  pred_vs_gt.mp4            # visualization
```

## Few-shot / annotation-budget study

One run = one annotation budget: annotate `annotation_frames` (evenly spread),
auto-annotate & score everything else. To see how accuracy scales with annotation
effort, sweep the one knob — giving each run its own `work_dir` so its `report.md` and
`pred_vs_gt.mp4` are kept:

```bash
for N in 100 200 300 500; do
  sed "s/annotation_frames: .*/annotation_frames: $N/" configs/gas_valve.yaml > /tmp/cfg_$N.yaml
  python -m pipeline.cli all --config /tmp/cfg_$N.yaml
done
```

`work_dir: runs/gas_valve_{frames}` auto-labels each run by its budget, so this writes
`runs/gas_valve_100/report.md`, `runs/gas_valve_200/…`, etc. — each with its own
localization (with/without classes) report and comparison video. More annotated frames
= higher accuracy; the sweep shows how few you can get away with.

## Metrics

Evaluation reports, on the held-out val frames:

- **Standard detection** (class-aware Hungarian match): precision / recall / F1 /
  **accuracy** (= TP/(TP+FP+FN), Jaccard) at each IoU threshold, for the full pipeline
  (box **and** name) and for localization only (box, name ignored).
- **Compounded (naming lens)**: detection-recall, naming-accuracy-on-found, and
  end-to-end, each micro (per box) and macro (per class).

## Modules

| File | Role |
|---|---|
| `pipeline/config.py` | load + resolve YAML |
| `pipeline/dataset.py` | frame listing, split, budget subset, YOLO dataset build |
| `pipeline/phase1.py` | detector train + inference |
| `pipeline/phase2.py` | DINOv2 embedder, reference build, kNN naming |
| `pipeline/metrics.py` | IoU + P/R/F1/accuracy accumulator |
| `pipeline/visualize.py` | pred-vs-GT video |
| `pipeline/pipeline.py` | orchestrator (prepare→train→infer→evaluate→visualize) |
| `pipeline/cli.py` | command-line entry point |

## Documentation

Design docs and evaluation reports live in [docs/](docs/):

| Doc | What |
|---|---|
| [docs/plan.md](docs/plan.md) | Overall strategy and approach menu (Phase 1 & 2) |
| [docs/phase2_report.md](docs/phase2_report.md) | Phase 2 (naming) study — DINOv2 kNN |
| [docs/phase1_report.md](docs/phase1_report.md) | Phase 1 (localization) — detector training & recall |
| [docs/e2e_report.md](docs/e2e_report.md) | End-to-end Phase 1 → Phase 2 metrics |
