# Semi-Automated Annotation Pipeline (YOLO multi-class)

A single fine-tuned multi-class YOLO detector turns a few annotated frames into
auto-labels for the rest — it localizes **and** names objects in one shot:

```
frame ─▶ [multi-class YOLO detector → boxes + class names]
```

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

Copy `configs/doosan_swivel.yaml`, edit the `data:` paths and (optionally)
`project.root` (`auto` = the repo dir containing `configs/`). Key knobs:

| Field | Meaning |
|---|---|
| `project.steps` | which steps `all` runs — a list from `prepare, train, infer, evaluate, visualize`; drop any to skip it (e.g. omit `visualize` to skip the video). Running a step directly on the CLI ignores this and always runs. |
| `data.sampling` | how the annotated (training) set is picked — `even` (spread evenly), `random`, or `greedy` (below) |
| `data.annotation_frames` | **the annotation budget** — how many frames you annotate (spread evenly/randomly) to train the detector; used when `sampling: even|random`. Every other frame is auto-annotated & scored. |
| `data.samples_per_class` | per-class sample floor — used when `sampling: greedy` instead of `annotation_frames` |
| `data.eval_frames` | how many of the remaining frames to score (`-1` = all remaining) |
| `data.monitor_val` | small held-out subset for the detector's per-epoch validation |
| `phase1.model` | `yolo11l` or `yolo11x` base weights to fine-tune |
| `infer.conf` | detector confidence (lower = more recall, more review load) |
| `eval.iou_thresholds` | IoU thresholds for P/R/F1/accuracy |

## Run

```bash
CFG=configs/doosan_swivel.yaml
python -m pipeline.cli prepare    --config $CFG   # build the multi-class YOLO dataset
python -m pipeline.cli train      --config $CFG   # train the detector
python -m pipeline.cli infer      --config $CFG   # detect + name the val frames
python -m pipeline.cli evaluate   --config $CFG   # P/R/F1/accuracy @ IoU + per-class
python -m pipeline.cli visualize  --config $CFG   # stacked pred-vs-GT video
# or everything:
python -m pipeline.cli all        --config $CFG
```

Artifacts land under `project.work_dir`:

```
runs/<name>/
  dataset/                  # symlinked images + multi-class labels + data.yaml
  detector/train/weights/   # best.pt, last.pt
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
  sed "s/annotation_frames: .*/annotation_frames: $N/" configs/doosan_swivel.yaml > /tmp/cfg_$N.yaml
  python -m pipeline.cli all --config /tmp/cfg_$N.yaml
done
```

`work_dir: runs/doosan_swivel_{frames}_yolo` auto-labels each run by its budget, so
this writes `runs/doosan_swivel_100_yolo/report.md`, `runs/doosan_swivel_200_yolo/…`,
etc. — each with its own localization (with/without classes) report and comparison
video. More annotated frames = higher accuracy; the sweep shows how few you can get
away with.

## Greedy (coverage-optimal) frame selection

Instead of a fixed frame budget spread evenly, `sampling: greedy` picks the
*fewest* frames such that every class has at least `samples_per_class` sample
frames — most frames show several classes at once, so this reaches the same
per-class coverage with far fewer annotated frames than `even` sampling needs by
chance. Preview the trade-off first with `standalone_scripts/frame_budget_analysis.py`
(`--target N` or `--sweep LO:HI` to compare several N at once), then point a config
at it:

```yaml
data:
  sampling: greedy
  samples_per_class: 50   # instead of annotation_frames
```

`configs/doosan_swivel_greedy50.yaml` is a ready-made variant of
`configs/doosan_swivel.yaml` (`samples_per_class: 50`, same hyperparameters
otherwise) for comparing directly against the 1000-frame even-sampling run.

## Comparing checkpoints

`standalone_scripts/compare_detectors.py` runs `infer` + `evaluate` for two or more
YOLO checkpoints against the same config and writes a side-by-side comparison:

```bash
python standalone_scripts/compare_detectors.py \
    --config configs/doosan_swivel.yaml \
    --model trained=compare_model/doosan_trained_detector.pt \
    --model deployed=compare_model/doosan_deployed_detector.pt
```

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
| `pipeline/metrics.py` | IoU + P/R/F1/accuracy accumulator |
| `pipeline/visualize.py` | pred-vs-GT video |
| `pipeline/pipeline.py` | orchestrator (prepare→train→infer→evaluate→visualize) |
| `pipeline/cli.py` | command-line entry point |
