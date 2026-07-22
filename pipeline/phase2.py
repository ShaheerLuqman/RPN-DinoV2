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


def crop_boxes(img_bgr: np.ndarray, boxes_xyxy, min_px: int = 8):
    """Return (PIL RGB crops, kept_box_indices) skipping degenerate/tiny boxes."""
    H, W = img_bgr.shape[:2]
    crops, keep = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
        xi2, yi2 = min(W, int(round(x2))), min(H, int(round(y2)))
        if xi2 - xi1 >= min_px and yi2 - yi1 >= min_px:
            rgb = cv2.cvtColor(img_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2RGB)
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


def build_reference(cfg, embedder: Embedder, ref_stems: list[str], rebuild: bool = False):
    """Embed the GT crops of the reference frames -> (Xr, yr), cached under work_dir."""
    cache = cfg.reference_cache
    if cache.exists() and not rebuild:
        d = np.load(cache)
        return d["Xr"], d["yr"]
    images_dir, ext = cfg.images_dir, cfg.data.image_ext
    crops, labels = [], []
    for k, s in enumerate(ref_stems):
        img = cv2.imread(str(images_dir / f"{s}.{ext}"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_gt(images_dir / f"{s}.txt", W, H)
        cc, keep = crop_boxes(img, [(g[1], g[2], g[3], g[4]) for g in gt], cfg.phase2.min_box_px)
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
