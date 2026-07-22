# Semi-Automated Annotation of Industrial Workstation Frames

## 1. Problem Statement

We have a large set of frames from industrial-workstation footage that must be
annotated for **object detection (bounding boxes)**. Manual annotation of the
full set is infeasible, so we want a system that learns from a small
hand-labeled seed set and annotates the rest, with humans reviewing rather than
drawing from scratch.

### Key figures

| Item | Value |
|---|---|
| Total frames to annotate | ~5,000 |
| Frames we can label manually | ~300 |
| Number of classes | ~30 (variable — classes can be added/removed) |
| Example classes | bins (multiple types), hands, tools, screws, etc. |
| Annotation type needed | Bounding boxes |
| Frame source | Multiple videos; sampled frames are **non-consecutive** |
| Compute available | Multi-GPU / cluster |

### Constraints and their consequences

- **Non-consecutive frames** → frame-to-frame tracking / label propagation does
  **not** apply. Treat the 5,000 frames as effectively independent stills.
  (Optional lever: if the source videos are retained, neighboring frames around a
  target can be pulled, labeled once, and propagated with SAM 3.1 — kept in
  reserve, not in the main path.)
- **Boxes only** → no mask management; box-native detectors can be used directly.
  When a mask-producing model (e.g., SAM) is used, take the mask's bounding box.
- **Variable class count** → favors approaches where adding a class does **not**
  require full retraining (embedding/exemplar methods).
- **Multi-GPU cluster** → we can run two label sources in parallel and use their
  agreement as a quality signal; heavy tiled inference is cheap.

---

## 2. Core Strategy

The reliable pattern is an iterative **fine-tune → pseudo-label → human-review →
retrain** loop (self-training), *not* "let a VLM label everything." 300 frames
is not 300 examples — each frame holds many labeled instances, so common classes
already have thousands of examples. The real difficulty is the **long tail**
(rare classes) and **tiny objects** (screws), not the common classes.

### The loop

1. **Annotate the seed set strategically** — cover all ~30 classes, over-sampling
   rare and hard ones.
2. **Fine-tune a detector** (RT-DETR or Ultralytics YOLO-L/X, since compute is
   ample).
3. **Run inference on the remaining frames**, split predictions by confidence:
   - High confidence → auto-accept.
   - Mid confidence → human review queue (correcting a box is ~5–10x faster than
     drawing one).
   - Expected-but-missing detections → flag (missed objects can't be recovered
     later).
4. **Retrain** on the enlarged set. Repeat 2–3 rounds until per-class accuracy
   plateaus.

> **Non-negotiable:** always human-review pseudo-labels. Blindly trusting them
> compounds errors — noise from round 1 poisons round 2.

---

## 3. Upgraded Loop: Two Label Sources That Vote

With a cluster, run **two complementary label sources** on all unlabeled frames
and use their **agreement** as the confidence signal:

- **Source A — Fine-tuned detector (YOLO / RT-DETR):** owns fine-grained
  decisions and the common classes; fast and accurate once trained.
- **Source B — Open-vocabulary / exemplar model (SAM 3):** owns recall and acts
  as a second opinion; strong on generic concepts.

### Routing

| Condition | Action |
|---|---|
| Both agree (overlapping box, same class, high conf.) | Auto-accept |
| Disagree / only one fires / mid confidence | Human review queue |
| Neither fires where objects were expected | Flag (recall failure) |

The **disagreement set doubles as an active-learning selector**: frames where the
two models fight are the most informative ones to add to the next manual round.
With GPUs to spare, add test-time augmentation or a small detector ensemble and
treat ensemble disagreement the same way (cheap uncertainty estimation).

---

## 4. Class-by-Class Reality Check

The ~30 classes are **not** equally hard. Plan effort accordingly.

| Class type | Difficulty | Approach |
|---|---|---|
| **Hands** | Easy | Text prompt / generic detector works out of the box |
| **Common bins, big tools** | Easy–Medium | Fine-tuned detector handles well |
| **Fine-grained (bin type A vs. B, tool variants)** | Hard | Text prompts **fail**; use **image-exemplar** prompting (SAM 3) or embedding kNN; fine-tuned model as tiebreaker |
| **Rare / long-tail classes** | Hard | Embedding-nearest-prototype with 3–5 exemplar crops; YOLO is starved here |
| **Screws / tiny objects** | Hardest | **SAHI tiling** at train + inference; stricter review; never auto-accept blindly |

### On open-vocabulary limits

Open-vocab models are excellent for concepts *inside their training vocabulary*
(e.g., "hand") but noisy/unreliable for domain-specific and fine-grained objects.
A documented example: prompting an open-vocab detector on a specialized dataset
returned >9,000 detections of which only ~70 were the real target. Use these
models as **proposal generators and a QA layer**, not as the final fine-grained
labeler.

---

## 5. Small Objects: Screws

Standard detectors downsample small objects into oblivion before the detection
head sees them. Fix with **SAHI (Slicing Aided Hyper Inference)**:

- Slice each frame into overlapping tiles (e.g., 640×640, ~0.2 overlap).
- Detect at full resolution per tile, then merge with NMS.
- Reported AP gains ≈ 6.8–14.5% depending on detector/config.
- Apply at **both** inference and fine-tuning (slicing-aided fine-tuning).
- Works with YOLO or transformer detectors, no architecture change.

Keep screws on a separate, stricter review track — their pseudo-labels are the
least trustworthy, so eyeball them even when both sources agree.

---

## 6. Classification via Embeddings (Few-Shot / Metric Learning)

An alternative/complementary technique for **classifying** a region:

1. Get a region of interest (ROI).
2. Embed the crop with a strong backbone (**DINOv2/DINOv3**, or CLIP for text
   prompts).
3. Build a per-class prototype (mean embedding) or keep all exemplars (kNN).
4. For a new ROI, assign the nearest class by cosine similarity.

### What it solves and what it doesn't

- **Solves classification** ("what class is this crop?").
- **Does NOT solve localization** — it assumes the ROI already exists. The box
  still has to come from somewhere (see §7).

### Why it's valuable here

- **Long tail / fine-grained:** works with 3–5 clean exemplars where a detection
  head learns nothing.
- **Variable class count:** adding a class is free — drop in a few exemplar
  embeddings, no retraining. (Directly matches our changing class list.)
- **QA layer:** embed a YOLO-predicted crop, check similarity to the assigned
  class prototype; low similarity → flag likely error.

### Failure modes to plan for

- Fine-grained blur — coarse distinctions are easy, subtle ones may not separate;
  **test empirically** on your variants.
- Crop sensitivity — embeddings shift with box tightness/background/scale; keep
  exemplar crops consistent; prefer **kNN (k≈5)** over a single averaged prototype.
- No "unknown" by default — add a **rejection threshold** (min similarity, or
  small top-1-vs-top-2 margin → send to review).

> **Note:** SAM 3's image-exemplar prompting is essentially this technique
> productized (proposal + embedding-match in one model). Try it before building a
> custom SAM-proposals + DINO-embeddings + kNN stack.

### Existing asset: the EDA_TOOL DINOv2 pipeline (reuse for Phase 2)

We already have a working DINOv2 embedding pipeline in
`EDA_TOOL/EDA_intra_class_variation/` that empirically **clusters crops of the
same class together**. This is the concrete starting point for Phase 2
(assign-class-to-bbox). What it already gives us, reusable as-is:

| Step | Script | What it does | Reuse for Phase 2 |
|---|---|---|---|
| Crop | `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Crops bboxes per class from YOLO `.txt` labels → `cropped_imgs_by_class/<class>/*.jpg` | Build the reference library from the seed set's boxes |
| Embed | `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | `facebook/dinov2-base`, **768d**, mean-pool of `last_hidden_state`; saves `embeddings_dinov2.npy` + image-list | **Identical** embedder for reference crops *and* new query crops |
| Cluster | `postannotation_scripts/3. clustering_of_classes_embeddings.py` | Per-class **DBSCAN on L2-normalized embeddings / cosine distance**, auto-tuned eps (k-NN percentile) | Produces per-class **sub-cluster centroids** = ready-made multi-prototypes |

**Intended Phase-2 flow:** build the embedding reference from the ~300 seed
frames' boxes → for each Phase-1 localized crop, embed it with the *same* DINOv2
(768d, mean-pool, L2-normalize) → assign the nearest class by cosine similarity.

**The gap to close — this tool clusters, it does not yet classify.** Today it does
*intra-class* clustering (find sub-groups/outliers *within* one class) for EDA. It
has **no cross-class assignment** step ("which of the 30 classes is this new
crop?"). Phase 2 must add:

- **Reference index:** per class, keep either the DBSCAN **sub-cluster centroids**
  (preferred — natural multi-prototype, matches §6's "kNN over single averaged
  prototype" guidance) or all exemplar embeddings for kNN.
- **Assignment:** cosine-similarity nearest-prototype / kNN (k≈5) over the index.
- **Rejection threshold:** min-similarity or top1-vs-top2 margin → send to review
  (the "unknown" handling from §6; the EDA clustering has no notion of it).

**Consistency requirement:** query crops must be embedded with the *exact* config
the reference used — `dinov2-base`, mean-pool, L2-normalize, and the same crop
convention (tightness/padding). The tool's crop pipeline defines that convention;
Phase-1 crops must match it or embeddings shift (the crop-sensitivity failure mode
in §6). Note the tool's *pre-annotation* path uses a different 1536d
`[CLS||avg_patches]` embedding — Phase 2 uses the **768d crop** path, not that one.

---

## 7. Region Proposal (Where the ROIs Come From)

> **Now the active workstream.** Phase 2 (naming a given box) is validated — see
> `phase2_report.md`: ~96% overall / ~89% fair-average with enough exemplars, and a
> confidence gate that auto-labels ~95% of boxes at ~99% accuracy. The remaining job
> is **Phase 1 — produce the boxes.** The quick table below is the menu; **§12 is the
> full Phase-1 plan, ranked, with a first experiment.**

Classification needs boxes. Options, from classic to modern:

| Proposer | Notes |
|---|---|
| **RPN (Faster R-CNN stage 1)** | Classic class-agnostic proposer; weak on tiny objects and on out-of-distribution objects; only "class-agnostic" within its training distribution |
| **SAM 3 / SAM 2 "segment everything" → boxes** | Modern RPN replacement; handles novel/arbitrary objects far better; mask→box is trivial (boxes-only need). **Recommended proposer.** |
| **Open-vocab detectors as objectness (Grounding DINO / YOLO-World / OWLv2)** | Promptable proposers; high recall, noisy precision — fine, since the embedding classifier makes the class decision |
| **DETR-family (RT-DETR / DINO-DETR)** | Replace RPN + anchors with learned object queries; if fine-tuned, the decoder *is* the proposer — stronger than classic RPN |

### The design fork

- **Separate proposer + embedding classifier (decoupled):** best when the class
  list changes (swap prototypes, keep proposer fixed) and for rare classes.
- **Fine-tuned one-stage detector / DETR (joint):** localization + classification
  in one model; faster and usually more accurate on well-represented classes, but
  retrain when classes change.

These are **not** mutually exclusive — run both branches and vote (§3).

For **screws**, combine any proposer with **SAHI tiling** (propose on
full-resolution tiles, not the downsampled full frame) — the single biggest fix
for tiny-object recall.

---

## 8. How Many Frames / Instances Per Class

It's **instances** (individual boxes) that matter, not frames.

| Instances per class | Expectation |
|---|---|
| ~50 | Bare minimum for *any* signal; OK for first pseudo-label pass only |
| ~150–300 | Practical sweet spot for a usable fine-tuned detector in a constrained domain |
| ~500+ | Common classes become genuinely solid |
| 1,000+ | Diminishing returns for easy classes; still worth it for hard ones |

What matters as much as raw count:

- **Diversity beats volume** — varied angles/lighting/occlusion/background teach
  more than near-identical crops. Non-consecutive multi-video frames are ideal.
- **Hard cases carry weight** — occluded, blurry, cluttered, partially-out-of-frame.
- **Balance** — if bins have 2,000 instances and a tool has 30, the model ignores
  the tool. Over-sample or augment rare classes.

**Action before training:** count instances per class in the seed set. Anything
under ~50 is a red flag — spend extra manual effort there before trusting the
model to auto-annotate it.

---

## 9. Recommended Tool Stack

| Need | Tool |
|---|---|
| Annotation + review UI | CVAT or Roboflow (both have SAM-assisted labeling + review queues) |
| Auto-label → train automation | Autodistill |
| Target detector | Ultralytics YOLO (v11/v12) or RT-DETR |
| Small objects | SAHI (train + inference) |
| Class-agnostic proposals | SAM 3.1 "segment everything" → boxes |
| Few-shot / fine-grained classification | SAM 3 image-exemplar prompting, or DINOv3 + kNN — **existing DINOv2 embedder + per-class centroids in `EDA_TOOL/EDA_intra_class_variation/` (see §6); add cross-class nearest-prototype assignment** |
| Object tracking (only if using source videos) | ByteTrack / BoT-SORT, or SAM 3.1 |

---

## 10. Concrete Schedule

Spend the 300-frame budget **across rounds**, not all at once. Pull ~50 frames
aside at the start as a fixed validation set.

- **Round 0** — Label ~100 frames maximizing class coverage (every class present;
  rare/hard classes over-sampled). Fine-tune detector. Build the SAM 3 exemplar
  library (3–5 clean crops per fine-grained/rare class).
- **Round 1** — Dual-source inference on all unlabeled frames → auto-accept
  agreements, review the disagreement queue. Use disagreements to select the next
  ~100 manual frames. Retrain.
- **Round 2** — Repeat. Auto-accept coverage should now be high; review queue
  small.
- **Round 3 (mop-up)** — Focused manual labeling on whatever still lags (usually
  screws and the rarest bins/tools).

**Stopping rule:** stop adding rounds when per-class AP on the held-out set stops
improving. Watch **per-class AP for the rare classes**, not the overall number
(which common classes dominate).

---

## 11. Summary of Recommendations

1. Train a detector on the seed set, use it to pre-annotate, **review** the output,
   retrain — iterate 2–3 rounds.
2. Run **two label sources** (fine-tuned detector + SAM 3 exemplar/open-vocab) and
   vote; agreement → auto-accept, disagreement → review + active learning.
3. Use **embedding/exemplar classification** for the long tail and fine-grained
   classes, and as a QA layer — it also handles the changing class list without
   retraining.
4. Use **SAM 3 "segment everything" → boxes** (or a fine-tuned RT-DETR decoder) as
   the region proposer rather than a classic RPN.
5. Apply **SAHI tiling** everywhere small objects (screws) appear.
6. Track **instances per class**, not frames; anything under ~50 needs more manual
   labeling before auto-annotation is trustworthy.

---

## 12. Phase 1 — Localization (Region Proposal): Deep Dive

Expands §7 now that Phase 2 is validated. The two phases are decoupled:

```
frame ──▶ [Phase 1: find objects → class-agnostic boxes] ──▶ crops
                                                              │
                                                              ▼
                                          [Phase 2: DINOv2 kNN → class name]
```

### 12.1 What Phase 1 actually has to do

Because Phase 2 does the naming, **Phase 1 only needs to find objects, not classify
them.** That reframes the problem into a much easier one:

- **Class-agnostic.** One job: "is there an object here, and where?" Not "which of 30
  classes." This is stable to the changing class list — adding class 31 does **not**
  require retraining the proposer (its job is unchanged).
- **Recall-first, precision-second.** Missed objects are unrecoverable (nobody drew a
  box, so no reviewer sees it); false boxes are cheap — Phase 2's **rejection
  threshold** (min cosine similarity / small top-1-vs-top-2 margin) drops proposals
  that match no class prototype, and the rest go to the review queue. So Phase 1
  should be tuned to **over-propose**.
- **The binding difficulties are the same two as everywhere:** tiny objects (screws)
  and getting boxes on the rare classes — not the common bins/hands.

> **Design consequence:** a high-recall class-agnostic proposer + Phase-2 embedding
> classifier + Phase-2 rejection is the natural decoupled stack. Precision junk is
> filtered downstream, so we can push recall hard.

### 12.2 Candidate approaches

| # | Approach | How it localizes | Training? | Strengths | Weaknesses / risks | Fit here |
|---|---|---|---|---|---|---|
| A | **Fine-tuned single-class detector** (RT-DETR or YOLO-L/X, all GT boxes collapsed to one "object" class) | Learned detector head/queries | Yes (we already have the boxes) | Reuses our labels; one-class is easy to learn; fast; strong on in-distribution objects; **stable to class-list changes** | Won't propose object *shapes* unseen in the seed; needs seed to cover scales | **Primary proposer** |
| B | **SAM 2 "segment everything" → boxes** | Automatic mask generation, mask→box | No | No training; handles novel/arbitrary objects; very high recall; mask→box trivial (boxes-only) | Over-/under-segments (splits one object into parts or merges several); floods proposals; slow; no objectness ranking for *our* notion of "object" | **Recall backstop / cold start / novel objects** |
| C | **Open-vocab detector** (Grounding DINO / YOLO-World / OWLv2), prompt "object"/"bin"/"hand"/"tool" or the class names | Promptable detection | No (optional fine-tune) | Zero-shot; promptable; can exploit class names; decent recall | Noisy precision; fails on fine-grained/domain objects (the >9k-detections example); threshold-sensitive | **Second opinion / QA, not final** |
| D | **Classic RPN** (Faster R-CNN stage 1) — the project namesake | Anchor + objectness head | Yes | Purpose-built class-agnostic proposer; well understood | Weak on tiny + out-of-distribution; "class-agnostic" only within training dist; generally **dominated by A** | Baseline / reference |
| E | **DETR object queries** (RT-DETR / DINO-DETR decoder as proposer) | Learned queries, set prediction (no NMS) | Yes | Strong localization; no anchor/NMS tuning; clean boxes | Query count caps max objects/frame; can under-detect dense tiny objects | A, implemented as DETR |
| F | **Classical** (Selective Search / Edge Boxes) | Low-level grouping | No | Trivial, training-free | Low precision; weak on texture-poor industrial scenes | Baseline only |

**Cross-cutting for tiny objects (screws):** wrap *any* proposer in **SAHI tiling** —
propose on full-resolution overlapping tiles, merge with NMS — rather than on the
downsampled full frame. Keep screws on the separate stricter track (§5).

### 12.3 Recommended Phase-1 stack

1. **Primary — one-class RT-DETR** (or YOLO-L) fine-tuned on all current GT boxes
   collapsed to `object`. We now have 1,500+ annotated frames from the Phase-2 work,
   which is plenty for a single-class detector.
2. **Recall backstop — SAM 2 automatic masks → boxes**, filtered by size / aspect /
   containment and by Phase-2 embedding match (drop masks that resemble no class).
3. **SAHI tiling** over both, for screws / tiny parts.
4. **Merge** the two proposal sets (NMS / weighted box fusion) → crops → Phase-2
   classifier → Phase-2 rejection drops non-objects, low-confidence → review.
5. **Vote (§3):** boxes where the fine-tuned detector and SAM agree → auto-accept;
   disagreements → review queue **and** active-learning selector for the next manual
   round.

Decoupled (A/B + embeddings) is the default because the class list changes; keep a
**joint** fine-tuned multi-class detector as the *second voting source* for the common
classes, where it's fastest and most accurate.

### 12.4 How to measure Phase 1 (recall-first metrics)

Localization is scored by overlap (IoU), independent of the name:

- **Average Recall (AR@k)** and **Recall @ IoU 0.5** (and the stricter 0.5:0.95) — the
  headline: what fraction of true objects got a proposal box.
- **Per-class miss rate**, watching **rare + tiny** classes specifically — the overall
  number will be dominated by common objects and look deceptively good (same
  overall-vs-fair-average lesson as Phase 2).
- **Proposals per frame** — the cost Phase 2 + reviewers pay for recall.
- Precision / mAP is secondary here (downstream filtering handles false boxes), but
  track it so we know the review-queue size.

### 12.5 First experiment (mirror the Phase-2 methodology — it worked)

Reuse the **exact same frame split** as Phase 2 (held-out 1,000-frame query set) so
Phase-1 and Phase-2 numbers compose.

1. Build **class-agnostic GT**: map every box → single `object` class.
2. Train **one-class RT-DETR** (and a YOLO-L baseline) on the seed/reference frames.
3. Evaluate **AR@100 / Recall@IoU0.5 + per-class miss rate** on the 1,000 held-out
   frames. Break out screws and the rare hand-poses.
4. **Baselines, same metric:** SAM 2 everything→boxes (with/without SAHI); open-vocab
   prompt ("object" and the class names).
5. **End-to-end:** run Phase 1 → Phase 2 and measure compounded (box **and** name)
   accuracy against the **upper bound we already have** — Phase 2 on GT boxes
   (~96% / ~89%). The gap is pure localization loss.
6. Decide the operating point: proposer + confidence/NMS thresholds that maximize
   recall at a review-queue size we can afford.

**Open questions to resolve in the experiment:**
- Does the one-class detector generalize to rare-class *shapes* it saw few times, or
  does SAM carry recall there?
- SAHI tile size / overlap for screws vs. inference cost.
- Where to set Phase-2 rejection so it prunes SAM's false proposals without dropping
  real rare objects.