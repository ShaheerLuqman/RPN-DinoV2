# Faster R-CNN Proposer — Architecture

This document explains the Faster R-CNN detector used in **Phase 1** of the pipeline,
as implemented in [`pipeline/frcnn.py`](../pipeline/frcnn.py). It is the trained
alternative to the training-free proposers (SAM / FastSAM / YOLOE).

## The big picture: it only *localizes*

The most important thing about this FRCNN is that it is a **class-agnostic proposer**.
It is built with `num_classes=2` — literally just **"object" vs "background"**
([`frcnn.py:28`](../pipeline/frcnn.py#L28)), and every ground-truth label is forced to
`1` at [`frcnn.py:86`](../pipeline/frcnn.py#L86). It answers *"where are the objects?"*
and hands those boxes to **Phase 2** (DINOv2 kNN) to answer *"what is each one?"*. That
is why it is interchangeable with the SAM/YOLOE proposers — they all emit boxes in the
same interface.

## The data flow (two-stage detector)

```
   image (1024px)
        │
        ▼
┌─────────────────────┐
│ 1. Backbone + FPN   │  ResNet-101 → multi-scale feature maps (P2…P6)
│    (ResNet-101-FPN) │
└─────────┬───────────┘
          │  shared feature pyramid
          ▼
┌─────────────────────┐
│ 2. RPN              │  slides anchors over every FPN level,
│  (Region Proposal   │  scores each "object / not", nudges the box
│   Network)          │  → a few hundred candidate regions
└─────────┬───────────┘
          │  proposals
          ▼
┌─────────────────────┐
│ 3. RoI Head         │  RoIAlign crops each proposal from the
│  (RoIAlign → MLP →  │  feature map to a fixed grid, an MLP refines it
│   FastRCNNPredictor)│  → 2-way class score + box regression
└─────────┬───────────┘
          ▼
   boxes + "object" scores  →  Phase 2 naming
```

### 1. Backbone + FPN — the feature extractor

A **ResNet** (50, 50-v2, or **101** per the config) pretrained on ImageNet, wrapped in
a **Feature Pyramid Network**. FPN produces feature maps at several resolutions so the
detector sees small *and* large objects well.

- ResNet-50 paths use torchvision's prebuilt `fasterrcnn_resnet50_fpn(_v2)`
  ([`frcnn.py:32-42`](../pipeline/frcnn.py#L32-L42)).
- ResNet-101 builds the backbone explicitly via
  `resnet_fpn_backbone(..., trainable_layers=3)`
  ([`frcnn.py:47`](../pipeline/frcnn.py#L47)) — **only the top 3 ResNet stages are
  fine-tuned**; the early layers stay frozen at their ImageNet weights (standard
  transfer-learning move: saves compute and avoids overfitting a few thousand frames).

Input sizing: `min_size=imgsz` (1024) and `max_size=max(1333, imgsz)`
([`frcnn.py:31,39,48`](../pipeline/frcnn.py#L31)) — torchvision rescales each image into
that band before the backbone.

### 2. RPN — Region Proposal Network (stage 1)

Slides a small conv over every FPN level with a set of **anchors** (boxes of different
sizes / aspect ratios) at each location. For each anchor it predicts (a) an
**objectness score** and (b) a **box refinement**. It keeps the top-scoring,
NMS-filtered anchors as **proposals**. This is what makes it "Faster" R-CNN — proposals
come from the network itself, not a separate algorithm. The pipeline uses torchvision's
default RPN (not customized).

### 3. RoI Head — the second stage

For each proposal, **RoIAlign** crops the corresponding region from the FPN features
into a fixed-size grid, an MLP box-head processes it, and the **`FastRCNNPredictor`**
outputs two things per RoI: a **class score** (here just object/background) and a
**refined box** ([`frcnn.py:40-41`](../pipeline/frcnn.py#L40-L41)). This is the head that
gets swapped to set `num_classes=2` — the pretrained COCO predictor (81 classes) is
replaced with a fresh 2-class one so it re-learns "object" for this data.

## Training ([`frcnn.py:90-114`](../pipeline/frcnn.py#L90-L114))

- **Data:** frames with ≥1 box (FRCNN needs non-empty targets,
  [`frcnn.py:74`](../pipeline/frcnn.py#L74)); images as RGB float tensors, all labels = 1.
- **Loss:** in training mode the model returns a **dict of 4 losses** — RPN
  classification + RPN regression + RoI classification + RoI regression — and the trainer
  optimizes their **sum** ([`frcnn.py:107`](../pipeline/frcnn.py#L107)). Both stages train
  jointly, end-to-end.
- **Optimizer:** SGD, lr 0.005, momentum 0.9, weight-decay 5e-4
  ([`frcnn.py:97-99`](../pipeline/frcnn.py#L97-L99)); batch = `phase1.batch // 4`
  (FRCNN is memory-heavy).

## Inference ([`frcnn.py:127-146`](../pipeline/frcnn.py#L127-L146))

Eval mode returns `boxes` + `scores`; the proposer thresholds by `infer.conf` and caps
at `max_det`, processing 4 images at a time because the model is heavy. The output tuple
`(boxes, scores, image, shape)` matches the other proposers' interface, so Phase 2 does
not care which proposer produced the boxes.

## Why this design (vs. the multiclass YOLO path)

Decoupling **localize (FRCNN) → name (DINOv2 kNN)** means adding a new class needs **no
detector retraining** — just more reference crops in Phase 2. A single-class "find any
object" detector is also easier to train well on few annotated frames than a full
multi-way detector.
