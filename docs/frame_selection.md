# Frame Selection for Annotation — Problem & Implementation

> Status: **implemented and running** (`frame_extraction/select_frames.py`), validated
> end-to-end on two datasets, and currently in a train-and-compare validation run against
> even-sampling (§5). Every design decision below was made explicitly with the user, not
> assumed.
> Scope: **upstream of everything in [README_pipeline.md](../README_pipeline.md)** — this is about
> which frames get pulled out of raw video *before* any labels exist, not how the
> already-labeled pool is later subset for training (`data.sampling: even|random|greedy|manifest`,
> see below for why that's a different problem — and how the two now connect via the new
> `manifest` sampling mode).
>
> Constraint that shapes everything below: this runs **before initial model training** —
> no checkpoint, no labels, no annotated samples of any kind exist yet. Nothing in this
> design may depend on a trained model or a labeled example (that rules out uncertainty
> sampling, active learning, and anything scene-graph/relation-based — all of them need
> either a model's output or existing boxes to work with; they belong to a later,
> post-bootstrap stage, not this one).

## Executive summary

Given raw video or an already-extracted frame folder and **no labels or trained model**,
`frame_extraction/select_frames.py` picks a small, non-redundant subset of frames for
annotation instead of the previous fixed-stride/evenly-spaced approach. It embeds every
frame with a frozen DINOv2 backbone, collapses near-duplicates using an adaptive
distance-based rule (not a fixed time window, so it doesn't mis-size across task cycles
that run at different speeds), and then greedily picks the most mutually-different frames
(k-center / farthest-point sampling) until a target budget (default 5%) is reached. The
result is written both as a manifest (JSON) and as actual staged files (symlinked/hardlinked
where possible), and can be fed straight into the training pipeline via a new
`data.sampling: manifest` mode for direct comparison against even-sampling at the same
frame count. Run twice already — `doosan_tackweld_processed` (4,416 → 221 frames) and
`doosan_swivel` (22,889 → 1,144 frames) — and a head-to-head detector-training comparison
on the latter is in progress (§5).

## 1. The problem

Raw input is video (`standalone_scripts/download_videos_s3_timestamp.py` pulls it,
`standalone_scripts/infer_videos.py` and `prepare_dataset.py` consume it later) — tens of
thousands of frames per recording at typical capture rates. Only a small slice of those
frames can be hand-annotated. Today the frames that get sent to annotators are picked by
simple positional rules (fixed stride / evenly spaced), which has two failure modes at once:

- **Redundant:** long idle/static stretches (robot paused, part sitting untouched, camera
  not moving) produce dozens of near-identical frames. Evenly-spaced sampling still burns
  budget on these — an annotator draws the same box in the same place 15 times in a row.
- **Under-covered:** short, fast, or rare events (a state transition, brief occlusion, a
  part flipped or a tool swapped) can fall entirely between two sample points and never
  get annotated at all, even though they're exactly the frames the model most needs to
  see. This is the current process's biggest structural risk, not just an efficiency
  loss — a class that's real but poorly represented in the annotated set trains a
  detector that's poor on it too. `summary.md`'s Doosan Swivel results already show class
  *presence* isn't the whole story (`torque_gun_placed` had 20,348 boxes yet 0.113 recall
  under one checkpoint) — undersampling could make a similar failure originate upstream,
  in what never got labeled in the first place, not just in training.

**Goal:** pick a frame subset per video that is *maximally informative per frame
annotated* — every frame in the subset should look meaningfully different from every
other frame in it, so annotator effort buys new information rather than repetition, and
rare/brief events aren't more likely to be skipped than long steady-state ones. All of
this has to work with **zero prior labels or model output** — the only thing available is
the raw video itself.

**Non-goal:** this is not about how many frames to annotate in total, in general — that's
the existing annotation-budget question (`data.annotation_frames` / `data.samples_per_class`,
`standalone_scripts/frame_budget_analysis.py`). The budget used so far is a fixed 5%
(§4), but the method itself isn't meant to be the last word on budget-setting.

## 2. Why this is a different problem from the existing `greedy` sampler

The pipeline already has a "pick a good subset of frames" mechanism —
`pipeline.dataset.greedy_cover`, used by `data.sampling: greedy` (see
[README_pipeline.md](../README_pipeline.md#greedy-coverage-optimal-frame-selection) and
`standalone_scripts/frame_budget_analysis.py`). It's tempting to reach for the same tool
here, but it solves a different problem:

| | `greedy_cover` (existing) | Frame extraction for annotation (this doc) |
|---|---|---|
| Input | Already-**labeled** frames | Raw, **unlabeled** video |
| "Coverage" measured by | Class counts from YOLO `.txt` labels | Embedding-space novelty — no labels to count |
| Picks frames to... | Subset an existing labeled pool down for *training* | Decide which frames get *sent to annotators at all* |
| Optimizes for | ≥N labeled examples per known class | Minimizing redundancy / maximizing coverage over an unknown label space |

The two aren't unrelated — both are instances of "greedily maximize marginal coverage,"
just over a different notion of coverage (class counts vs. embedding-space distance).
`select_frames.py`'s k-center step (§4) is the unsupervised analogue: same greedy shape,
embedding distance instead of class counts. The two are now also connected downstream —
`pipeline.dataset.manifest_annotate_split` (new, §5) lets a `select_frames.py` manifest
feed straight into the same `prepare`/`train`/`infer`/`evaluate` pipeline `greedy_cover`
already powers.

## 3. The signal: embedding distance

Two cheaper alternatives were considered and set aside:

- **Pixel/motion difference** (raw pixel diff, histogram distance, optical flow) is
  free and needs no model to run, but conflates *camera/lighting noise* with
  *task-relevant change*, and is purely local — it can't tell that two frames from two
  different idle periods, far apart in time, are near-duplicates of *each other*, only
  that each differs from its own immediate neighbors. It under-fixes the redundancy
  problem from §1.
- **Detector-uncertainty / scene-graph signals** would be stronger, but both need a
  model's output (a trained checkpoint's confidence, or boxes to build relations from) —
  unavailable by construction at this stage (see the constraint box up top). Out of
  scope here; they're the natural next step *after* a bootstrap model exists, not part
  of getting there.

**Chosen signal: learned embedding distance**, via DINOv2, frozen, no fine-tuning and no
labels required (§4 has the shipped model size/metric).

## 4. Implementation — `frame_extraction/select_frames.py`

Every parameter below is what's actually shipped and running, not just planned.

| Decision | Shipped choice | Why |
|---|---|---|
| **Datasets run so far** | `datasets/doosan_tackweld_processed` (4,416 frames → 596 bucket representatives → 221 selected) and `datasets/doosan_swivel` (22,889 frames → 5,332 bucket representatives → 1,144 selected) | Both are already-labeled, `prepare_dataset.py`-staged flat frame folders — chosen over the original raw-video target (`marmon_station_1`) so the same source doubles as real input *and*, via the new `manifest` sampling mode, lets the downstream train-and-compare validation (§5) run immediately without waiting on a fresh annotation pass. |
| **Selection scope** | Global — pool all groups' frames into one embedding space, one budget, one k-center pass | Catches redundancy across videos of the same station/task, not just within one video. For a flat, already-staged frame folder (both runs so far) this is moot in practice — it's a single group — but the code supports multi-video pooling for a future raw-video run. |
| **Embedding backbone** | DINOv2 ViT-B/14 (`facebook/dinov2-base`, via HF `transformers`), frozen, CLS token, L2-normalized | Balanced speed/quality default; fits the repo's own DINOv2-based naming-stage history (`summary.md`) even though the current pipeline is YOLO-only post strip-down. `transformers` added to `requirements.txt`. |
| **Decode rate** | Every native frame (no pre-stride) | Full thoroughness — a brief event can't fall between sample points if every frame is a candidate. All redundancy reduction happens downstream (pre-bucketing + k-center), not by skipping frames upfront. |
| **Quality pre-filter** (shipped defaults, all overridable via CLI/config) | Blur: Laplacian variance < `60.0` · Exposure: mean brightness outside `[15, 240]` · Occlusion: brightness std < `5.0` (near-uniform frame ⇒ lens covered) | Classical CV, no model needed. Exact thresholds are implementation-time tuning constants, not empirically validated yet — both real runs so far had `quality_dropped: 0`, i.e. these defaults haven't actually been exercised/tested against real bad frames yet. |
| **Temporal pre-bucketing** | **Adaptive, by embedding distance (default δ = 0.10 cosine), not a fixed time window.** Walk frames in temporal order; keep a frame as a new bucket representative whenever its embedding distance from the last kept representative exceeds δ. | A fixed-duration window (e.g. "always 1 second") was explicitly rejected: task cycles run at different speeds — a fast/efficient cycle compresses real events into less wall-clock time than a slow one with delays, so a fixed-time window either splits one fast event across two buckets or wastefully over-buckets a slow, static one. Sizing by *how much the content actually changed* instead of by wall-clock time makes bucket size self-adjust to cycle speed automatically. |
| **Embedding cache** | Yes — one `.npz` per run under `<run-dir>/cache/`, keyed by frame id, loaded once and shared across all groups, saved incrementally (every `10 * batch_size` newly-embedded frames, atomic tmp-file swap) | Embedding is the expensive step; caching means changing the budget or re-running later is nearly free. Incremental saves specifically so an interrupted run — and runs *have* been interrupted in practice, both a plain terminal interruption early on and the memory incident below — only loses a few batches, not the whole run. |
| **Streaming, not batch-eager, processing** *(added after a real incident, not originally planned)* | Frames flow through quality-filter → embed → adaptive-bucket in bounded `--batch-size` chunks; **at most one batch of raw frames is ever held in memory**, never the whole dataset. | The first version eagerly loaded every frame into a Python list before processing (`iter_sources()` returning `list(...)`). Fine at 4,416 frames; at 22,889 frames it drove the process to **~24GB RSS with the system down to 0.8GB free** — a near-OOM caught live and fixed by rewriting the frame source as lazy generators plus a streaming `process_group`/`CachingEmbedder` pair. Verified byte-identical output on the smoke test before and after the rewrite (this was a memory-profile fix, not a behavior change). |
| **Diversity/selection algorithm** | k-center greedy (farthest-point sampling) over the pooled post-bucketing representative set, cosine distance | No extra hyperparameter besides the budget, degrades gracefully as the budget changes — simpler than tuning an ε for a facility-location/set-cover variant. Reuses the same greedy-selection *shape* as `pipeline/dataset.py`'s `greedy_cover`, over embedding distance instead of class counts. |
| **Distance metric** | Cosine distance throughout (pre-bucketing and k-center both) | Standard for comparing deep embeddings; scale-invariant. Embeddings are L2-normalized once at embed time so cosine distance is just `1 - dot product`. |
| **Frame budget** | 5% of candidates (post quality-filter), globally, via `--keep-fraction` (default `0.05`) | Fixed value given directly, not swept — both real runs used the default 5%. |
| **Input types supported** | Both raw video files (`--videos-dir`) and pre-extracted frame folders (`--images-dir`) | Frame folders can be either flat/sequential (`prepare_dataset.py`-style `000001.jpg`, both real runs) or per-video-named (`<video>_frame_000205.jpg`), auto-detected via the same `FRAME_RE` pattern `prepare_dataset.py` uses. |
| **Output** *(expanded beyond the original plan)* | **Manifest** (JSON: params, counts, `selected[].{id, pick_order}`) **and staged files** — symlinked/hardlinked from source (falls back through `pipeline.dataset._link_or_copy`'s symlink→hardlink→copy chain; confirmed hardlink-only on this machine, no Developer Mode) for frame-folder input, written from the decoded frame for video input. Sibling YOLO labels + `classes.txt` are staged alongside when the input was already labeled. `--no-stage` skips this. | Originally planned as manifest-only "traceability, staging deferred" — staging was added once requested, since a manifest alone isn't handoff-ready. |
| **Run layout** *(refined after initial version)* | Everything for one run lands under a single `<run-dir>` (default `runs/frame_selection/<source-name>/`): `cache/`, `selected/` (staged files), `selected.json` (manifest) | Matches the same "one folder per run" convention `pipeline/config.py`'s `work_dir` already uses — originally the cache lived in a separate `runs/frame_selection_cache/` tree, consolidated on request. |
| **Config-driven runs** *(added after initial version)* | `--config configs/*_frame_selection.yaml` — same convention as `pipeline/cli.py`: the YAML is the whole interface when given (copy/edit for a new run rather than mixing in CLI flags), `project.root: auto` resolves the same way `pipeline/config.py`'s does | `configs/doosan_tackweld_frame_selection.yaml` and `configs/doosan_swivel_frame_selection.yaml` are the two real configs used so far. |
| **Script location** | `frame_extraction/select_frames.py` (new top-level dir — not `standalone_scripts/`) | User's explicit choice. |

### Pipeline order (as shipped)

```
for each group (a video, or a recognized frame-folder group) — streamed, not preloaded:
  for each --batch-size chunk of frames, in temporal order:
    1. read the frame
    2. quality pre-filter (blur + exposure + occlusion) -> drop unusable frames
    3. embed survivors with frozen DINOv2 ViT-B/14, via the shared on-disk cache
    4. adaptive pre-bucketing (cosine-distance threshold delta, streamed across
       batches) -> keep this group's bucket representatives

pool every group's representatives together (global scope)
5. k-center greedy (cosine distance) over the pooled set until keep_fraction of
   total candidate frames is selected
6. write manifest.json (selected frame identifiers, pick order, params, counts)
7. stage the selected frames as files (symlink/hardlink + labels + classes.txt),
   unless --no-stage
```

### Why two dedup stages, not one

The `doosan_swivel` run's summary line is easy to misread as one selection step:

```
22,889 total frames -> 0 dropped (quality) -> 22,889 candidates
                     -> 5,332 after pre-bucketing -> 1,144 selected (5.0% of total)
```

The 5,332 is an *intermediate* pool, not part of the final selection — it's the output of
step 4 (adaptive pre-bucketing), which is a different, cheaper kind of dedup than step 5
(k-center). They catch different kinds of redundancy, which is why both exist:

**Stage 1 — adaptive pre-bucketing (22,889 → 5,332): cheap, local, sequential.** Walk the
candidate frames in temporal order. Keep the first as a "representative." For every frame
after that, compare it only to the *last kept representative* — not to any other frame:

```
frame 1  -> kept (first frame, always a representative)
frame 2  -> distance from frame 1 = 0.02   -> too similar, discard
frame 3  -> distance from frame 1 = 0.03   -> too similar, discard
   ...
frame 47 -> distance from frame 1 = 0.14   -> different enough (> delta=0.10) -> new representative
frame 48 -> distance from frame 47 = 0.01  -> too similar, discard
   ...
```

A long idle/static stretch (nothing moving) can collapse to a single representative across
hundreds of frames, while a fast-changing stretch mints a new representative every few
frames — that's the "adaptive" property from §4's bucketing row: bucket size isn't a fixed
frame/second count, it's however many frames it takes for content to drift past delta.
22,889 → 5,332 means ~17,500 of this dataset's frames were near-duplicates of the frame
immediately next to them in time — a real measurement of how much of the raw footage is
redundant, not an arbitrary knob.

**Stage 2 — k-center greedy (5,332 → 1,144): expensive, global, all-pairs.** This is the
actual final-selection step (§4's "diversity/selection algorithm" row), and it only runs
on the 5,332 survivors, not all 22,889. It repeatedly picks whichever remaining frame is
*farthest*, in embedding space, from everything already picked — comparing every candidate
against every other candidate, not just temporal neighbors.

**Why not do this in one pass:** the two stages catch different redundancy. Bucketing only
compares a frame to its immediate predecessor, so it's blind to two far-apart moments that
happen to look alike (e.g. two separate idle periods on opposite ends of the video look
identical to *each other*, but neither would trigger the other as "the last kept
representative"). k-center's all-pairs comparison catches exactly that case — but running
it on the raw 22,889 (instead of the pre-thinned 5,332) would pay that all-pairs cost over
a pool that's still ~80% near-duplicate frames, for no benefit bucketing didn't already
capture more cheaply. Bucketing is a fast first pass that removes "obviously the same as
the frame right before it"; k-center is the actual diversity-maximizing selection over
what's left.

## 5. Validation

**Plan (unchanged from the original design):** train the existing pipeline's detector
twice at the same frame budget — once on frames picked by this method, once on frames
picked by plain even-sampling — and compare `report.md` metrics (especially
**macro**/per-class, given `summary.md`'s finding that per-class fairness is where budget
choices show up first). This reuses the existing few-shot/annotation-budget study
machinery ([README_pipeline.md](../README_pipeline.md#few-shot--annotation-budget-study))
— just feeding it a different frame set at the same N instead of `even`. This is the
actual acceptance test for the method — not a proxy metric computed without ever training
anything.

**How it's wired (new since the original plan):** rather than requiring a fresh
annotation round-trip, the pipeline gained a fourth sampling mode —
`data.sampling: manifest` (`pipeline.dataset.manifest_annotate_split`) — that reads a
`select_frames.py` manifest directly and uses exactly those frames as the training set,
since `doosan_swivel` (the dataset used) happens to already be fully labeled. This is
strictly a validation convenience: it lets today's already-labeled dataset stand in for
the "annotate the selected frames" step so the comparison can run now, but it does **not**
change what the method itself depends on (§0 constraint) — selection still runs with zero
labels; only the *evaluation* of that selection reuses labels that happen to already exist.

**Result: even-sampling won, on every metric measured.** Two configs, identical in every
hyperparameter except the training-set source:

- `configs/doosan_swivel_frameselect1144.yaml` — `sampling: manifest`, trained on the
  1,144 frames `select_frames.py` picked (`runs/frame_selection/doosan_swivel/selected.json`)
- `configs/doosan_swivel_even1144.yaml` — `sampling: even`, trained on 1,144 evenly-spread
  frames — the control, same count, same everything else

Both ran `prepare → train → infer → evaluate` end-to-end (YOLO11l, 15 epochs, scoring the
remaining ~21,745 frames each):

| Metric | Frame-selection | Even-sampling | Winner |
|---|---:|---:|---|
| Localization P/R/F1/Acc @0.5 | 0.973 / 0.950 / 0.961 / 0.925 | 0.978 / 0.956 / 0.967 / 0.936 | even |
| Localization P/R/F1/Acc @0.75 | 0.936 / 0.915 / 0.925 / 0.861 | 0.948 / 0.926 / 0.937 / 0.881 | even |
| End-to-end P/R/F1/Acc @0.5 | 0.962 / 0.940 / 0.951 / 0.906 | 0.973 / 0.951 / 0.962 / 0.927 | even |
| End-to-end P/R/F1/Acc @0.75 | 0.928 / 0.906 / 0.917 / 0.847 | 0.943 / 0.922 / 0.932 / 0.874 | even |
| Detection recall — micro / macro | 0.954 / 0.759 | 0.959 / 0.808 | even |
| Naming acc\|found — micro / macro | 0.984 / 0.782 | 0.991 / 0.845 | even |
| **End-to-end — micro / macro** | 0.938 / **0.677** | 0.950 / **0.742** | even, macro gap +0.065 |
| Classes covered in training (of 55) | 54 | 55 | even |
| False positives | 18,886 | 14,581 | even (~23% fewer) |
| Mean IoU | 0.909 | 0.914 | even |

Full reports: `runs/20260818_061001_doosan_swivel_frameselect1144_yolo/*_report.md` and
`runs/20260818_061049_doosan_swivel_even1144_yolo/*_report.md`.

**Root cause, visible directly in the reports, not just inferred:** frame-selection's
1,144 picks contained **zero** examples of `tighten_fit_1_m_view_wrench` (0 GT boxes in the
training-frames breakdown); even-sampling's picks happened to include 10. The detector
trained on frame-selection's picks then scored 0.006 recall on that class at eval time (163
real instances in the eval set) — essentially never able to recognize it, because it never
saw one. Even-sampling's detector, with only 10 training examples of the same rare class,
still did marginally better (though still bad — 0.059 recall). This is exactly the §7
"visual diversity ≠ task-relevant diversity" risk playing out concretely, not
hypothetically: a class defined by a subtle wrench-in-hand state looks embedding-similar to
its neighboring frames even though it's a distinct, task-relevant label, so the diversity
method's redundancy-collapsing logic (adaptive pre-bucketing, §4) can end up discarding the
only examples of it — the opposite of the intended effect from §1's "under-covered" failure
mode.

## 6. Resolved / no-longer-open questions

- **Quality-filter thresholds and the pre-bucketing δ** now have shipped defaults (§4),
  but are still *implementation defaults, not empirically validated* — both real runs so
  far show `quality_dropped: 0`, meaning these thresholds haven't actually been exercised
  against real bad frames yet. Treat them as easy-to-override starting points, not tuned
  values.
- **Manifest schema** is now settled (JSON: `source`, `model`, `params`, `counts`,
  `selected[].{id, pick_order}`), not a remaining unknown.
- **Visual diversity ≠ task-relevant diversity — confirmed, not just a risk.** §5's
  head-to-head training comparison shows this concretely: even-sampling beat the
  embedding-diversity method on every metric measured, and the gap traces to a specific,
  identifiable cause (a rare fine-grained-state class collapsed out of the training set
  entirely). This is no longer a hypothetical worth flagging for later — it's the headline
  result of the first real validation run.

## 7. Open questions still not fully closed

- **Is this a fixable parameter problem, or a structural limitation of the method?** The
  §5 result doesn't distinguish between "embedding-diversity selection is fundamentally
  the wrong idea for this kind of dataset" and "the bucketing/budget defaults are too
  aggressive at collapsing rare-but-visually-similar frames." A smaller bucket delta
  (less aggressive collapsing), a larger keep-fraction, or a hybrid method (e.g. diversity
  selection plus a `greedy_cover`-style per-class floor layered on top, closer to what
  `data.sampling: greedy` already does for the labeled-pool case) could plausibly close
  the gap — none of that has been tried yet.
- **One dataset, one run.** This is a single comparison on `doosan_swivel` at one budget
  (5%/1,144 frames) with one seed. It's a real, concrete result, not a fluke traceable to
  a specific rare class — but it's still one data point, not a settled verdict on the
  method in general.
- **Per-video vs. cross-video pooling** hasn't actually been exercised yet — the datasets
  used for training comparison were single-group flat frame folders. The `marmon_station_1`
  run (multi-video, raw video input) is the first real test of that path, separate from
  and not yet informing this validation result.
- **Quality-filter thresholds never actually filtered anything** on real data so far
  (see §6) — worth deliberately checking against a frame folder/video known to contain
  blurry or occluded frames before trusting the filter in production.

## 8. Next step

**Do not promote this method to the default annotation-prep step as-is** — §5's result is
a real regression against the simple baseline it was meant to beat, not a win. Before any
further use on a labeling budget that matters, one of two things needs to happen: either
(a) try a fix for the identified cause — e.g. a smaller bucket delta, or layering a
`greedy_cover`-style per-class floor on top of the diversity selection so rare classes
can't be fully collapsed out — and re-run this same head-to-head comparison to see if it
closes the gap, or (b) treat this as evidence the pure-diversity approach needs a different
core idea for class-rich, fine-grained-state datasets like this one, not just a tuning
pass.
