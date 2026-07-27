"""Phase 2 — naming. DINOv2 crop embeddings + kNN vote against a reference library
built from the annotated frames' ground-truth crops.

Embedding config matches the validated Phase-2 study: mean-pool of last_hidden_state,
L2-normalized (768-d for dinov2-base).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel


class Embedder:
    def __init__(self, model_name: str, device: str = "0"):
        self.device = (f"cuda:{device}" if torch.cuda.is_available() and str(device) != "cpu"
                       else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, crops: list[Image.Image], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            inp = self.proc(images=crops[i:i + batch], return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            feats = self.model(**inp).last_hidden_state.mean(dim=1)
            out.append(feats.float().cpu().numpy())
            if (i // batch) % 25 == 0 and len(crops) > batch:
                print(f"    embedded {min(i + batch, len(crops))}/{len(crops)}", flush=True)
        if not out:
            return np.zeros((0, self.model.config.hidden_size), np.float32)
        return normalize(np.vstack(out).astype(np.float32), norm="l2")


class BoxMasker:
    """Box-prompted SAM2 -> per-box binary mask (ultralytics SAM).

    Mirrors the proven acp-agentic-workflow `Sam2Masker`: prompt the whole frame
    with all boxes at once, then read `r.masks.data` aligned to the input box order.
    Used by Phase 2 to suppress background before DINOv2 embeds each crop."""

    def __init__(self, weights, device: str = "0"):
        from ultralytics import SAM
        self.weights = Path(weights)
        self.model = SAM(str(weights))
        self.device = device
        self._mm = None                         # lazy multimask predictor

    def masks_for(self, source, boxes_xyxy):
        """Return a list aligned to boxes_xyxy; each entry is a HxW uint8 mask or None."""
        if len(boxes_xyxy) == 0:
            return []
        r = self.model.predict(source, bboxes=np.asarray(boxes_xyxy, np.float32),
                               device=self.device, verbose=False)[0]
        if r.masks is None:
            return [None] * len(boxes_xyxy)
        data = r.masks.data.cpu().numpy()
        return [(data[i] > 0).astype(np.uint8) if i < len(data) else None
                for i in range(len(boxes_xyxy))]

    def _multimask_predictor(self):
        """Lazily build the low-level SAM predictor used for per-box multimask output.
        The high-level `SAM.predict` NMS-merges masks across all box prompts, which
        loses per-box attribution; the predictor lets us prompt one box at a time."""
        if self._mm is None:
            name = self.weights.name.lower()
            if "sam2" in name:
                from ultralytics.models.sam import SAM2Predictor as P
            else:
                from ultralytics.models.sam import Predictor as P
            self._mm = P(overrides=dict(task="segment", mode="predict",
                                        model=str(self.weights), save=False,
                                        verbose=False, imgsz=1024, device=self.device))
        return self._mm

    def multi_masks_for(self, source, boxes_xyxy):
        """Per-box multimask: return a list aligned to boxes_xyxy, each entry a list of
        (HxW uint8 mask, quality_score) candidates for that box — SAM's whole/part/subpart
        hierarchy. Reveals composite objects (e.g. a hand+tool box yielding several masks).
        Each box is prompted separately so masks stay attributed to their box."""
        if len(boxes_xyxy) == 0:
            return []
        pred = self._multimask_predictor()
        img = cv2.imread(str(source)) if isinstance(source, (str, Path)) else source
        pred.set_image(img)
        out = []
        for b in boxes_xyxy:
            r = pred(bboxes=np.asarray([b], np.float32), multimask_output=True)[0]
            if r.masks is None:
                out.append([])
                continue
            data = r.masks.data.cpu().numpy()
            scores = (r.boxes.conf.cpu().numpy() if r.boxes is not None
                      else np.ones(len(data), np.float32))
            out.append([((data[i] > 0).astype(np.uint8), float(scores[i]))
                        for i in range(len(data))])
        return out


def crop_boxes(img_bgr: np.ndarray, boxes_xyxy, min_px: int = 8, masks=None):
    """Return (PIL RGB crops, kept_box_indices) skipping degenerate/tiny boxes.

    When `masks` is given (list aligned to boxes_xyxy, each a HxW mask or None), the
    background inside a crop is zeroed before conversion — the acp background-suppression
    recipe, so DINOv2 embeds the object shape rather than the box's surroundings."""
    H, W = img_bgr.shape[:2]
    crops, keep = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
        xi2, yi2 = min(W, int(round(x2))), min(H, int(round(y2)))
        if xi2 - xi1 >= min_px and yi2 - yi1 >= min_px:
            sub = img_bgr[yi1:yi2, xi1:xi2].copy()
            m = None if masks is None else masks[i]
            if m is not None:
                if m.shape[:2] != (H, W):        # SAM masks may return at model res
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                m = m[yi1:yi2, xi1:xi2]
                if m.shape[:2] == sub.shape[:2]:
                    sub[m == 0] = 0
            rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))
            keep.append(i)
    return crops, keep


def read_gt(txt: Path, W: int, H: int):
    """(class_id, x1, y1, x2, y2) in pixels from a YOLO label file."""
    out = []
    for line in Path(txt).read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            c = int(float(p[0])); xc, yc, w, h = map(float, p[1:])
            out.append((c, (xc - w / 2) * W, (yc - h / 2) * H,
                        (xc + w / 2) * W, (yc + h / 2) * H))
    return out


def build_reference(cfg, embedder: Embedder, ref_stems: list[str], rebuild: bool = False,
                    masker: "BoxMasker | None" = None):
    """Embed the GT crops of the reference frames -> (Xr, yr), cached under work_dir.

    When `masker` is given, background is SAM-suppressed before embedding — this must
    match how query crops are built at inference so reference and query live in the
    same (masked or unmasked) embedding space."""
    cache = cfg.reference_cache
    if cache.exists() and not rebuild:
        d = np.load(cache)
        return d["Xr"], d["yr"]
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    crops, labels = [], []
    for k, s in enumerate(ref_stems):
        path = images_dir / f"{s}.{ext}"
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_gt(images_dir / f"{s}.txt", W, H)
        boxes = [(g[1], g[2], g[3], g[4]) for g in gt]
        masks = masker.masks_for(str(path), boxes) if masker is not None else None
        cc, keep = crop_boxes(img, boxes, cfg.phase2.min_box_px, masks=masks)
        crops += cc
        labels += [gt[i][0] for i in keep]
        if (k + 1) % 200 == 0:
            print(f"  reference crops {len(crops)} ({k + 1}/{len(ref_stems)})", flush=True)
    Xr = embedder.embed(crops, cfg.phase2.batch)
    yr = np.array(labels, int)
    np.savez_compressed(cache, Xr=Xr, yr=yr)
    return Xr, yr


def classify_knn(Xq: np.ndarray, Xr: np.ndarray, yr: np.ndarray, k: int = 5):
    """Distance-weighted kNN vote.
    Returns (pred_class_ids, vote_confidence[0..1], top_similarity[0..1]) where
    top_similarity is the cosine similarity to the single nearest reference crop —
    the signal for "does this crop resemble any known object" (used by the reject gate)."""
    if len(Xq) == 0:
        return np.array([], int), np.array([], float), np.array([], float)
    k = min(k, len(Xr))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xr)
    dist, idx = nn.kneighbors(Xq)
    sims = 1.0 - dist
    preds, conf, topsim = [], [], []
    for row_s, row_i in zip(sims, idx):
        votes = defaultdict(float)
        for s, j in zip(row_s, row_i):
            votes[int(yr[j])] += max(s, 0.0)
        c = max(votes, key=votes.get)
        preds.append(c)
        conf.append(votes[c] / (row_s.clip(min=0).sum() + 1e-9))
        topsim.append(float(row_s.max()))
    return np.array(preds, int), np.array(conf, float), np.array(topsim, float)
