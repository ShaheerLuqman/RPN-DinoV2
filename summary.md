# ACP Auto Annotation — Summary of Attempts

## Attempt 1: Object Tracking using SAM (Video Annotation Propagation Pipeline, plan.md)

**Approach name:** Object tracking using SAM2, driven by **evenly spaced anchor frames**. Anchor frames (the manually labeled frames) were spaced evenly through the video, and propagation moved **forward and backward between each consecutive pair of anchor frames**, fusing the two directions to reduce drift.

**Goal:** Given a video and 20–30 manually labeled frames (bounding boxes with fixed object IDs), automatically annotate the full video while preserving object identity across all frames — handling occlusion, rotation, viewpoint change, and appearance change, while keeping expensive VLM usage limited to uncertain cases only.

**Proposed architecture:**
- **Primary engine:** SAM2 video propagation (box → mask on anchor/labeled frames, then propagate)
- **Identity layer:** Custom tracker (Kalman filter + IoU + appearance embedding matching, Hungarian assignment)
- **Uncertainty resolver:** VLM called only on flagged/uncertain frames or events
- **Correction loop:** Human-in-the-loop review for the worst/most ambiguous frames, with corrections fed back as new anchors
- **Output:** masks + boxes + track_id + confidence + occlusion flag, per frame

**Core design principle:** Treat this as semi-supervised video object segmentation/tracking, not frame-by-frame detection. Labeled frames are "identity anchors" — object `1` must stay object `1` everywhere; IDs are never invented or swapped for known objects.

**Key pipeline stages:**
1. Extract frames, load/validate labels, build anchor timeline (sorted labeled frame indices per object)
2. Convert anchor boxes → masks via SAM2 image predictor (with quality checks: bbox IoU vs human box ≥0.70, mask/box area ratio 0.25–1.50)
3. Segment video between consecutive anchors; run SAM2 propagation **forward and backward** within each segment (not one long forward pass) to reduce drift
4. Fuse forward/backward masks via agreement IoU thresholds (≥0.70 confident, 0.40–0.70 medium, <0.40 uncertain)
5. Identity tracker maintains per-object state (Kalman state, embeddings, occlusion status) and resolves multi-object assignment via weighted cost (mask IoU 0.30, box IoU 0.20, motion 0.20, appearance 0.20, shape 0.10)
6. Explicit occlusion states (visible/partially_occluded/fully_occluded/lost/recovered) — occluded objects keep their ID, never reassigned
7. Per-frame/per-event uncertainty scoring (weighted combo of SAM confidence, fwd/bwd IoU, motion jump, area change, appearance distance, overlap) drives escalation: auto-accept <0.30, low-confidence accept <0.60, VLM resolve <0.80, human review ≥0.80
8. VLM used only on grouped uncertain "events" (not isolated frames), with strict JSON output format, budget capped (~1–5% of frames), response caching
9. Human review limited to keyframes around ambiguous events (before/during/after), corrections become new anchors, pipeline reruns propagation locally around them

**Evaluation plan:** Box/Mask IoU, J&F (DAVIS-style), IDF1, HOTA, ID switches, track fragmentation, occlusion recovery accuracy, flag precision/recall for the uncertainty system, VLM cost/utility metrics, runtime/FPS/cost.

**Suggested acceptance targets:** Box IoU@0.5 ≥90%, Mean Box IoU ≥0.75, IDF1 ≥90%, ID switches ≤1/1000 frames, occlusion recovery ≥85%, review rate ≤2–5% of frames, VLM call rate ≤1–5% of frames.

**Build order specified:** box/mask utils → label loader → SAM2 anchor mask gen → SAM2 forward propagation → backward propagation+fusion → export → basic evaluator → tracker (Kalman+IoU) → appearance embeddings → uncertainty scoring → VLM resolver → human correction loop → full evaluation report. Explicitly: **do not start with the VLM** — it's the final escalation layer once the local system works.

**Reported benchmark (this approach, run on 500 frames / 150 anchors, later re-run on SAM3):** mean IoU 0.775, F1@0.5 0.946, IoU@0.5 0.981, IDF1 0.644, 30 ID switches, 2.56 fps.

**Limit hit:** This design tracks *classes*, not *object states*. It had no notion that a label like `swage_with_hand` is not a distinct physical object — it's the base object `swage` in a temporary state (holding something). Concretely, in `preview5555.mp4`, `swage_with_hand` and `gauge_with_hand` stayed "active" (kept being predicted/propagated) even after the hand had left the frame, because the tracker treated the composite label as its own independent track. This motivated the next attempt.

---

## Attempt 2: SAM2 → SAM3 migration

Swapped SAM2 for Meta's SAM3 (image prompter used in SAM1-style box→mask mode — measured better than best-of-3 mask selection; video propagator drives SAM3's tracker head). This was a mechanical model upgrade, not a design change, but became the substrate every later scene-graph experiment built on.

---

## Attempt 3: Relation-aware composite handling (first scene-graph idea)

**Fix, stated as an invariant:** track physical objects, not composite class names. A composite label (e.g. `swage_with_hand`) is only emitted while its defining spatial relation (hand overlaps base object) actually holds; it is never propagated or motion-predicted on its own once that relation breaks.

This introduced relation-scoring (a relation FSM + relation scorer) and an object ontology that folds composite states back to a base family.

**Target set for this phase:** ≥0.80 composite macro-F1, ≥0.80 box accuracy@0.5.

---

## Attempt 4: Generalize to full scene graphs

The single hard-coded "hand overlaps base" rule was generalized: every ordered pair of objects in a frame gets scored for 8 spatial predicates (`overlaps`, `inside`, `contains`, `near` + 4 directional). These are aggregated from ground-truth anchor frames into an `AggregatedSceneGraph` prior — a per-family-pair table of which relations are ever attested and how often.

The prior is used two ways:
1. **Box arbitration** — SAM3's forward and backward passes are both kept per object; whichever candidate (`fused` / `fwd` / `bwd`) best agrees with the prior is emitted.
2. **Output labeling** — the predicted graph is spoken only in the ground-truth vocabulary, and every edge is tagged `expected` / `unexpected` / `novel` / `missing` against the prior, giving a per-frame scene-graph consistency score.

Architecture doc was rewritten to "Final" at this point with explicit invariants: composites are graph edges/states, never independent tracks; human anchor geometry is authoritative; held-out eval labels never leak into anchoring/calibration.

---

## Attempt 5: Forward-only inference (matching the real eval constraint)

**Realization:** the actual task spec only allows seeding from the **first** frame of each validation sequence — no backward pass is legitimate, since there's no "future" ground truth to propagate from in the real deployment setting. This forced a redesign: seed once, propagate strictly forward, fill lost objects with Kalman motion prediction, and use the Attempt-4 prior only to *hold* the previous box when a forward-propagated box breaks an otherwise-expected relation ("scene-graph hold") — never to move a box forward.

The distilled, "faithful to the design doc" version of this became the reference architecture (purely geometric decision, no appearance embedding bank, no detection-free proposals — cheap and deterministic).

**Limit hit:** dense, forward-only SAM3 propagation with grid-point proposals is slow — on the order of hundreds of SAM forwards per frame.

---

## Attempt 6: Trading SAM3 for a fast open-vocabulary detector (YOLOE)

Replaced SAM3 propagation entirely with **YOLOE**, prompted *visually* (never with text) using per-class visual-prompt embeddings baked from the anchor crops. Scene-graph verification (from Attempt 4) did double duty: suppress "ghost" detections whose geometry disagrees with the prior, and rescue misses by relaxing the confidence floor for families the prior expects but that are currently absent.

**Result:** ~30x faster than the SAM3 forward/backward pipeline (1.9 min vs ~60 min per run) — but a real accuracy regression: YOLOE's visual-prompt head confuses near-identical composite classes (e.g. 10 `hand_w_screwdriver` variants, 3 part-on-white-fixture stages), producing wrong-class and ghost boxes.

---

## Attempt 7: Recovering accuracy on the fast path

Kept YOLOE as the fast region proposer but added: optional SAM2 masking to background-suppress the crop before embedding (skipped for small objects, where the box already tightly bounds the object), a DINOv2 **prototype** re-scorer on top of the YOLOE confidence, Weighted Box Fusion + containment cleanup to merge/dedupe overlapping proposals, and the Attempt-4 scene-graph verification pass. Goal: stay close to YOLOE's runtime while recovering most of the accuracy lost to class confusion.

---

## Attempt 8: Alternative accuracy path — SAM3 proposals + open-set re-ID

A parallel branch of experimentation, explicitly framed against Attempts 6/7: *"proposals themselves were unreliable — wrong-place boxes that no amount of DINOv2 re-ID could rescue."* Goes back to SAM3, but only for **class-agnostic region proposals** (a point-grid over the frame, detection-free, no class assigned) — cleaner, background-suppressed regions than YOLOE gave.

Classification is then fully separated out into an **open-set DINOv2 re-ID** step against a *multi-exemplar* per-class bank (kNN over anchor crops, not one averaged prototype like Attempt 7). The scene-graph prior gets a new job here too: a first pass keeps only high-confidence matches, then a second "demand recovery" pass accepts otherwise-rejected proposals for families the prior expects but that are missing, gated by geometric consistency. Finishes with temporal track voting (one class per track by appearance-weighted vote, gap-filling, ghost pruning).

---

## Attempt 9: Rethinking the primary signal — geometry over appearance

The biggest conceptual pivot, stated directly in the pipeline's own docstring: *"Earlier pipelines were appearance-first: propose a region, match its crop to a per-class DINOv2 bank, then bolt a scene graph on the end as a weak verifier. That is structurally wrong for this dataset."*

Two observations drove this:
- The per-class anchor folders contain **full-scene** YOLO labels (every object in frame, not just the anchor's own class) — i.e. co-occurrence and layout are available "for free" from data already being used.
- The hardest classes (`f1_FR/FL/BR/BL`, `f2_*`, `*_on_white_bin`) are defined by **position, not appearance** — their crops look nearly identical; what actually distinguishes them is where they sit relative to the fixtures/bins.

**New method:** auto-select stable, frequent classes (bins/fixtures) as **reference families**; per validation sequence, detect those references once and lock their boxes as a local coordinate frame; learn **relative-offset templates** (Δcx, Δcy, log scale-ratio) for every (class, reference) pair — camera-invariant, so they transfer across rigs/resolutions. Final classification blends `w_app * appearance_sim + w_geo * geometry_fit`, so appearance still disambiguates visually-distinct classes while geometry resolves the positional ones appearance can't. Entirely training-free / few-shot.

---

## Attempt 10: Speed pass on the geometry-primary approach

Same method as Attempt 9, with three throughput fixes: batch the SAM3 point-grid proposals into one `predict_inst` call instead of grid×grid sequential calls (~2-3x speedup on the proposal step); detect and lock static reference boxes (bins/fixtures) once per sequence instead of every frame; stride the expensive dense grid every N frames, relying on temporal gap-fill for moving classes on skipped frames while static classes just persist.

---

## Summary table

| Attempt | Approach | Primary signal | Key motivation for the change |
|---|---|---|---|
| 1 | SAM2 propagation + Kalman/IoU/appearance tracker | Mask propagation | Original design |
| 2 | → SAM3 | Mask propagation | Model upgrade |
| 3 | Relation-gated composites | Mask propagation + 1 hand-relation rule | Fix `*_with_hand` tracks surviving after hand leaves |
| 4 | Full scene graphs (8 predicates) | Mask propagation + learned relation prior | Generalize the one-off rule; use prior to arbitrate fwd/bwd boxes |
| 5 | Forward-only + "SG hold" | Mask propagation (fwd only) + prior | Match real eval constraint (1 seed frame, no future GT) |
| 6 | YOLOE visual-prompt detector | Detector confidence + prior verification | SAM3 dense-grid forward pass too slow (~60 min/run) |
| 7 | YOLOE + DINOv2 re-scoring + SAM2 masking | Detector + appearance prototype + prior | Recover accuracy YOLOE lost on near-identical composite classes |
| 8 | SAM3 class-agnostic proposals + open-set DINOv2 re-ID | Appearance (multi-exemplar) + prior | YOLOE proposals themselves were unreliable, not just re-ID |
| 9 | Relational few-shot (reference-anchored offsets) | Geometry (relative offset to locked references) + appearance | Position-defined classes are appearance-ambiguous; appearance-first is "structurally wrong" here |
| 10 | Fast relational | Same as 9, optimized | Runtime |

**The arc, in one line:** SAM-based mask propagation + classic tracker → relation-aware composite handling → full scene graphs → forward-only + scene-graph-guided refinement → detector-speed tradeoffs (YOLOE) → appearance+geometry fusion → geometry-as-primary-signal (training-free relational matching).

**Where this leaves things (as of the `fawad_v2` branch):**
- Attempts 1-10 all exist as separate runnable scripts — they read as a lab notebook of parallel experiments rather than a single pipeline that "won."
- No committed benchmark numbers exist in-repo for Attempts 6-10 (each writes its own accuracy/report file at run time, not checked in).
- The main branch never received any of this — it's still on the Attempt 1/2 (SAM2-era, pre-scene-graph) architecture.
- Natural next step: run Attempts 6-10 head-to-head on the same validation split and record accuracy + runtime in one place, since right now the comparison only lives in docstring prose.

---

*(More attempts/resources to be added as shared.)*

---
---

# RPN-DinoV2 — Research Summary

> Everything above this line is a different project (ACP auto-annotation /
> scene-graph tracking) and is unrelated to what follows. This section covers
> the research approaches explored for **RPN-DinoV2**'s auto-annotation
> pipeline.
>
> Numbers previously logged for these experiments (accuracy %, recall,
> counts) are considered **stale/obsolete** and are intentionally omitted —
> this is a record of *what was tried and why*, not current performance.

## The core design decision: decouple "find" from "name"

The founding idea was to split auto-annotation into two independent stages rather than one multi-class detector:

```
frame ─▶ [Phase 1: find objects → class-agnostic boxes] ─▶ crops ─▶ [Phase 2: DINOv2 kNN → class name]
```

- **Phase 1** only answers "is there an object here" — class-agnostic, recall-first, stable when the class list changes (no retraining needed to add class #31).
- **Phase 2** only answers "which of the ~27 classes is this crop" via DINOv2 embeddings + few-shot kNN — no detector training needed to add a class either, just a few more reference crops.

This decoupling is the thread that everything else hangs off.

## Phase 2 experiments — naming, tested first (against GT boxes so results are purity-of-naming, not box quality)

Tested against a reference set spread evenly across a recording (guaranteeing a minimum example count per class) and a held-out test split.

- **Three kNN comparison strategies tried:** one-average (single mean prototype per class), few-looks (multi-prototype via clustering, e.g. distinct camera angles), and all-examples/kNN vote (keep every reference crop, let the k nearest vote). Finding: no single method dominates everywhere — the vote method wins big on common, visually-distinct classes, but for several rare hand-pose classes the simpler one-average prototype did meaningfully better. This motivated the later idea of combining methods per class-group rather than picking one globally.
- **Confidence gate:** thresholding the kNN vote's own confidence lets the pipeline auto-accept the "sure" majority of boxes and route the uncertain remainder to human review — established as the practical deployment mechanism rather than trusting every prediction.
- **Annotation-budget sweep** (how many frames to hand-label before returns diminish):
  - A per-class floor (top up rare classes to a minimum frame count) reaches a good accuracy/label-effort tradeoff well before exhausting the full dataset — annotating everything was shown to not be worth it.
  - Plain even-sampling (no per-class floor) needs meaningfully more total frames to reach the same fair-per-class accuracy that a balanced floor reaches with fewer frames — balancing is the more label-efficient policy.
  - Exact-N-per-class balancing (capping *every* class, including common ones, to N examples) rescues rare classes dramatically but hurts common-class accuracy by starving their otherwise-abundant examples — concluded **not** the right default; better as a per-class-group blend (full example set for common classes, capped/boosted treatment for rare ones).
- **Background masking with SAM2** (mask out everything outside the box before embedding, testing whether background context helps or hurts naming): negligible effect on a dataset with tight boxes already near-ceiling accuracy, but a real fair-average improvement on a cluttered, many-class dataset — worth enabling only on cluttered/confusable datasets; off by default since it roughly doubles compute (an extra SAM2 pass per frame).
- **Conclusion:** naming works well for visually-distinct classes (bins, fixtures, parts); the unsolved tail is the "same hand doing a slightly different task" classes, which a single still crop structurally can't disambiguate — flagged as needing motion/position context or a stronger backbone, not more data.

## Phase 1 experiments — localization (region proposers)

Several proposer families were built and compared for the "find objects" stage:

| Proposer | Trained? | Role tried |
|---|---|---|
| **YOLO11‑L / YOLO11‑X, single-class** (all boxes collapsed to one generic `object` class) | Yes | **Chosen primary** — reuses existing labels, one class is easy to learn |
| **RT‑DETR** (swappable in for YOLO) | Yes | Alternative for tighter boxes (higher AR/IoU) |
| **Faster R‑CNN**, class-agnostic (object vs. background only — the project's namesake "RPN") | Yes | Baseline/reference — classic 2-stage proposer |
| **SAM / FastSAM**, "segment everything" → boxes | No (training-free) | Recall backstop for novel/OOD object shapes |
| **YOLOE** (prompt-free open-vocab) | No | Second opinion / zero-shot check |
| **YOLOE-visual** (visual-prompted from exemplar crops) | No | Fine-grained open-vocab alternative |

**Single-class vs. multi-class detector — the key design test:** trained both and compared. Single-class won decisively: it pools every GT box across all classes into one "objectness" signal, so rare classes still localize well, whereas a 27-way detector starves the rare-class heads and tends to drop low-confidence rare detections at threshold — the same long-tail problem Phase 2 has for naming, but avoidable here since Phase 1 doesn't need to name anything, only find.

**Result:** two operating points were compared — a very low confidence threshold that maximizes recall at the cost of more proposals per frame, vs. a higher-confidence "review-load" operating point that trades a little recall for far fewer proposals to review. Directionally: localization was found to be **not the bottleneck** — recall was high across essentially every class, including the classes Phase 2 struggles to *name* — which validated the find/name split (finding an object and naming it are different-difficulty problems). Box tightness (how well the box hugs the object, not just whether one exists) was the one metric still improving with more training, flagged as RT‑DETR's or more-epochs' job rather than a detection-recall problem.

## End-to-end (Phase 1 → Phase 2 wired together)

Detector → crop → DINOv2 kNN naming, scored against a held-out set, and compared against the Phase-2-alone upper bound (naming on perfect GT boxes).

Directionally: the end-to-end pipeline landed close to the GT-box naming ceiling on a per-box (micro) basis — i.e. localization costs almost nothing for the common classes. The gap that remained was larger on a per-class (macro) basis, and was diagnosed as a **naming** loss rather than a detection loss: predicted boxes are looser than hand-drawn GT boxes, and that looseness shifts the crop embedding enough to hurt the already-fragile fine-grained rare classes.

## Doosan Swivel — multi-class detector-as-namer experiment

A run (`doosan_swivel_1000_yolo11l_mc`) on the new Doosan Swivel dataset, testing the **joint** design (a single multi-class YOLO11‑L detector that finds *and* names in one shot) rather than the decoupled find→name approach used everywhere above. 1,000 annotated frames were used as the training/reference budget; the remaining **21,889 frames** (719,299 GT boxes across **55 classes**) were auto-annotated and scored — mirroring the real deployment scenario (annotate a small budget, auto-label the rest), not the earlier full-data "ceiling" runs.

**Dataset is materially harder/richer than the earlier `gas_valve` dataset:** 55 classes vs. 27, spanning not just objects (bins, fixtures, tools) but process *states* — tool-placement states (`torque_placed`, `wrench_placed`, `torque_empty`…), fixture-assembly states (`fit_2` … `fit_13_14_15_16`, several with distinct "M-view" camera-angle variants), and explicit **error/wrong states** (`wrong_fit_5_6`, `wrong_unit_on_fixture`, `wrong_t_fit_1_m_view`, …) that only differ from their "correct" counterpart by a subtle visual cue.

Two detector checkpoints were compared head-to-head with `compare_detectors.py` — the currently **deployed** checkpoint vs. a newer **trained** checkpoint:

**Localization (box only, class-agnostic):**

| IoU | Deployed — Precision / Recall / F1 / Acc | Trained — Precision / Recall / F1 / Acc |
|---|---|---|
| 0.50 | 0.988 / 0.902 / 0.943 / 0.892 | 0.988 / 0.977 / 0.982 / 0.965 |
| 0.75 | 0.962 / 0.879 / 0.918 / 0.849 | 0.970 / 0.959 / 0.964 / 0.931 |

**End-to-end (box + correct name):**

| IoU | Deployed — Precision / Recall / F1 / Acc | Trained — Precision / Recall / F1 / Acc |
|---|---|---|
| 0.50 | 0.987 / 0.901 / 0.942 / 0.890 | 0.986 / 0.975 / 0.981 / 0.962 |
| 0.75 | 0.961 / 0.878 / 0.918 / 0.848 | 0.969 / 0.958 / 0.963 / 0.929 |

**Compounded (naming lens), micro / macro:**

| Metric | Deployed | Trained (pure YOLO, w=1.0) |
|---|---|---|
| Detection recall @.5 | 0.903 / 0.897 | 0.978 / 0.846 |
| Naming acc \| found | 0.996 / 0.979 | 0.997 / 0.940 |
| End-to-end (found & named) | 0.900 / 0.879 | 0.974 / 0.814 |

**The non-obvious finding: the "trained" checkpoint is better in aggregate but worse for per-class fairness (macro), and *why* traces to a specific tradeoff, not general improvement/regression:**
- **Deployed's dominant failure was one outlier class:** `torque_gun_placed` — despite being one of the most frequent classes (20,348 GT boxes) — had catastrophic recall (**0.113**) in the deployed checkpoint, and `torque_placed`/`torque_empty` were also weak (recall 0.593 / 0.411). This single family of tool-placement classes dragged deployed's numbers down.
- **Trained fixed that family outright:** `torque_gun_placed` recall jumped to **0.998**, `torque_placed` to **0.998**, `torque_empty` to **1.000** — essentially solved.
- **But trained introduced a new, broader weak spot:** the `tighten_fit_*` family, which was solid under deployed (recall 0.82–0.98 across `tighten_fit_1_m_view`, `tighten_fit_2_3`, `tighten_fit_4_m_view`, `tighten_fit_5_6`, `tighten_fit_5_6_m_view`, `tighten_fit_7`, `tighten_fit_8`), collapsed under trained — recall fell to **0.052–0.60** across that same class family (worst: `tighten_fit_1_m_view_wrench` 0.052, `tighten_fit_8` 0.083, `tighten_fit_1_m_view` 0.120).
- **Net effect:** trained wins clearly on **micro** metrics (fixing a few very high-frequency classes moves the per-box average a lot) but *loses* on **macro** end-to-end (0.814 vs deployed's 0.879) because the regression now spans many classes (the whole `tighten_fit_*` family) instead of being concentrated in one. This is a real single-checkpoint-swap regression, not dataset noise — worth flagging before promoting "trained" to replace "deployed".
- **`wrong_unit_on_fixture` (an error-state class) remained a naming-specific weak point in both checkpoints:** deployed found it 83.4% of the time but named it correctly only 47.5% of those times (e2e 0.396); trained found it almost every time (99.9%) but still only named it correctly 82.0% of the time (e2e 0.819) — clearly the hardest class to *name* correctly once found, in both checkpoints, consistent with it being visually close to its "correct" counterpart (`unit_on_fixture`).

**YOLO ⇄ DINOv2 fusion weight sweep (trained checkpoint only — deployed run didn't have fusion enabled):**

| w (YOLO weight) | e2e micro | e2e macro |
|---:|---:|---:|
| 0.0 (pure DINOv2) | 0.966 | 0.750 |
| 0.3 | 0.970 | 0.780 |
| 0.5 | 0.973 | 0.801 |
| 0.7 | 0.974 | **0.814 ← best** |
| 1.0 (pure YOLO) | 0.974 | **0.814 ← best** |

False positives (8,568) were identical at every weight, as expected — fusion only changes *naming*, not which boxes are proposed. The sweep plateaus from w=0.7 upward, i.e. **the best blend found was effectively pure YOLO** — DINOv2 kNN naming did not add anything on top of the multi-class detector's own classification for this dataset/checkpoint (unlike the `gas_valve` Phase‑2 experiments, where different naming methods clearly won for different class subsets). Fusion is implemented and instrumented, but hasn't yet paid off here.

**Takeaway so far:** on this richer, state-heavy class taxonomy, checkpoint choice trades one class family's failure for another's — a caution against judging a retrained detector by micro/overall numbers alone. DINOv2 fusion, while cheap to compute (reuses cached predictions, no re-inference), hasn't yet beaten the detector's own classification on this dataset — open question is whether that changes with a different weighting scheme, more DINOv2 reference exemplars, or background masking (as tested earlier on `gas_valve`).


## Real trace-video validation — trained model as the production step-detector

Beyond the held-out benchmark split above, the **trained** checkpoint was run as `detector.py` directly against real task trace videos — i.e. in the shape it would actually run in deployment, not against the curated/pre-split frame dataset.

- **No frame extraction** — run straight against the video, not a pre-sampled/curated frame set.
- **No human validation or correction** — a fully automatic, uncorrected pass end-to-end.
- **5 trace videos tried**, each covering the full **48-step** task sequence — **240 step instances** in total across the 5 runs.
- **Result: performance was good.** Only **2 step instances** (out of 240) disagreed between checkpoints — cases where the **deployed** model marked a step as completed but the **trained** model did not.
- That's a **~0.8% (2/240) discrepancy rate**, which is considered low given this was a raw, uncurated, end-to-end run with no frame-selection help and no human-in-the-loop correction — the closest test yet to real deployment conditions rather than the benchmark split.

This is a useful complement to the checkpoint comparison above: the earlier per-class benchmark already flagged that "trained" trades one class family's recall for another's, so this real-video pass is exactly the kind of independent check needed before trusting the aggregate/micro benchmark numbers — and here it held up well, with only a small, specific number of missed step-completions rather than a broad regression.

## Whole-dataset benchmark vs. real trace videos — an open tension

The full head-to-head comparison (`comparison_trained_vs_deployed.md`) also reports the raw prediction volume behind the tables above:

| Metric | Trained | Deployed |
|---|---:|---:|
| Predicted boxes | 679,935 | 628,008 |
| False positives | 8,568 | 8,375 |
| Mean IoU (matched) | 0.9233 | 0.9093 |

Trained predicts noticeably more boxes overall for roughly the same false-positive count and a slightly tighter mean IoU — consistent with the earlier localization/end-to-end tables, where trained wins clearly on every **micro** (per-box) metric.

**The surprising part:** on the whole-dataset frame benchmark, trained looks like the better model almost everywhere except **macro** end-to-end (0.814 vs. deployed's 0.879). But on the actual real trace videos above, **deployed still came out ahead** — it caught step completions that trained missed, not the other way around. That's the opposite of what the micro numbers alone would predict.

This is flagged as an open question rather than a settled conclusion: it may mean the **macro** end-to-end number (where deployed already led, 0.879 vs 0.814) is the more honest predictor of real task performance than the micro number, since "did each of the 48 steps in a trace get detected" is inherently a per-step/per-class question, not a per-box one dominated by whichever classes have the most boxes in the dataset. If that holds up, the whole-dataset micro benchmark may be **overstating** how much better "trained" really is for this task, and macro end-to-end (or the trace-video test itself) should carry more weight than raw aggregate precision/recall when deciding which checkpoint to promote. Not yet confirmed — worth checking against more trace videos before treating it as settled.

## The arc, in one line

Two-stage decoupled design (class-agnostic localizer + few-shot DINOv2 namer) → validated naming first against GT boxes (kNN-vote naming, confidence gating, annotation-budget sweeps, background masking) → validated localization second (single-class beats multi-class detector for rare-class recall; SAM/FastSAM/YOLOE built as recall/second-opinion backstops; classic Faster R-CNN kept as the class-agnostic baseline) → wired end-to-end and found the macro/rare-class gap is a naming problem, not a detection one → now extending to a second dataset and testing whether fusing the multi-class detector's own opinion with DINOv2's vote recovers some of that rare-class naming gap.
