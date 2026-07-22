#!/usr/bin/env python3
"""
Phase-2 evaluation: DINOv2-embedding classification of GT bbox crops.

Isolates Phase 2 (assign-class-to-bbox) from Phase 1 (localization) by using
ground-truth boxes as the query crops. Builds a per-class embedding reference
from ~300 seed frames, then classifies crops from a disjoint held-out set and
scores predictions against the ground-truth class.

Embedder matches EDA_TOOL exactly: facebook/dinov2-base, mean-pool of
last_hidden_state, L2-normalized (768d).

Methods compared:
  centroid : one mean prototype per class (cosine nearest)
  knn      : k-NN over all reference exemplars (distance-weighted vote)
  cluster  : per-class DBSCAN sub-cluster centroids + noise singletons;
             assign to nearest prototype's class  (the "make clusters" approach)
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel


def log(msg):
    print(msg, flush=True)


def load_classes(path):
    names = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return names  # index == class_id


def list_frames(data_dir):
    stems = []
    for p in sorted(Path(data_dir).glob("*.png")):
        if p.with_suffix(".txt").exists():
            stems.append(p.stem)
    return stems


def read_boxes(txt_path):
    boxes = []
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cid, xc, yc, w, h))
    return boxes


def frame_class_index(data_dir, stems):
    """per_frame[i] = set of class ids in frame i;  by_class[c] = sorted frame idxs."""
    per_frame = []
    by_class = defaultdict(list)
    for i, s in enumerate(stems):
        cids = {b[0] for b in read_boxes(Path(data_dir) / f"{s}.txt")}
        per_frame.append(cids)
        for c in cids:
            by_class[c].append(i)
    return per_frame, by_class


def crop_frames(data_dir, stems, min_box_px):
    """Return (list[PIL crops], np.array class_ids, np.array box_area_px)."""
    crops, cids, areas = [], [], []
    skipped = 0
    for i, stem in enumerate(stems):
        img_path = Path(data_dir) / f"{stem}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for cid, xc, yc, bw, bh in read_boxes(Path(data_dir) / f"{stem}.txt"):
            x1 = int(round((xc - bw / 2) * W))
            y1 = int(round((yc - bh / 2) * H))
            x2 = int(round((xc + bw / 2) * W))
            y2 = int(round((yc + bh / 2) * H))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
                skipped += 1
                continue
            crop = img[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(crop))
            cids.append(cid)
            areas.append((x2 - x1) * (y2 - y1))
        if (i + 1) % 200 == 0:
            log(f"    cropped {i+1}/{len(stems)} frames, {len(crops)} boxes")
    return crops, np.array(cids), np.array(areas), skipped


@torch.no_grad()
def embed(crops, processor, model, device, batch=64):
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        b = crops[i:i + batch]
        inputs = processor(images=b, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                feats = model(**inputs).last_hidden_state.mean(dim=1)
        else:
            feats = model(**inputs).last_hidden_state.mean(dim=1)
        out.append(feats.float().cpu().numpy())
        if (i // batch) % 20 == 0:
            log(f"    embedded {min(i+batch, len(crops))}/{len(crops)} crops")
    if not out:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return normalize(np.vstack(out).astype(np.float32), norm="l2")


# ---- classifiers ----------------------------------------------------------

def cap_per_class(X, y, n, seed=0):
    """Downsample the reference set to exactly n exemplars per class.

    Classes with <= n exemplars are kept whole. For larger classes we keep n
    evenly-spaced exemplars across their (chronological) order so the kept crops
    still span the whole recording rather than clustering in one stretch.
    """
    if n <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) <= n:
            keep.extend(idx.tolist())
            continue
        pick = sorted(set(np.linspace(0, len(idx) - 1, n).round().astype(int).tolist()))
        if len(pick) < n:                        # rounding collapsed duplicates
            pool = [j for j in range(len(idx)) if j not in pick]
            pick = sorted(pick + rng.choice(pool, n - len(pick), replace=False).tolist())
        keep.extend(idx[pick].tolist())
    keep = np.array(sorted(keep))
    return X[keep], y[keep]


def build_centroids(X, y):
    protos, plabels = [], []
    for c in np.unique(y):
        protos.append(X[y == c].mean(axis=0))
        plabels.append(c)
    return normalize(np.array(protos), norm="l2"), np.array(plabels)


def build_cluster_prototypes(X, y, min_samples=2, percentile=90):
    protos, plabels = [], []
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 3:
            for v in Xc:
                protos.append(v); plabels.append(c)
            continue
        k = min(min_samples, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xc)
        d, _ = nn.kneighbors(Xc)
        eps = float(np.percentile(d[:, -1], percentile))
        eps = max(eps, 1e-3)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(Xc)
        for cl in set(labels):
            if cl == -1:
                for v in Xc[labels == -1]:      # noise -> singleton prototypes
                    protos.append(v); plabels.append(c)
            else:
                protos.append(Xc[labels == cl].mean(axis=0)); plabels.append(c)
    return normalize(np.array(protos), norm="l2"), np.array(plabels)


def predict_nearest_proto(Xq, protos, plabels):
    sims = Xq @ protos.T                 # cosine (all unit-norm)
    best = sims.argmax(axis=1)
    return plabels[best], sims[np.arange(len(Xq)), best]


def predict_knn(Xq, Xr, yr, k=5):
    k = min(k, len(Xr))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xr)
    dist, idx = nn.kneighbors(Xq)
    sims = 1.0 - dist
    preds, conf = [], []
    for row_sim, row_idx in zip(sims, idx):
        votes = defaultdict(float)
        for s, j in zip(row_sim, row_idx):
            votes[yr[j]] += max(s, 0.0)
        c = max(votes, key=votes.get)
        preds.append(c)
        conf.append(votes[c] / (row_sim.clip(min=0).sum() + 1e-9))
    return np.array(preds), np.array(conf)


# ---- metrics --------------------------------------------------------------

def evaluate(name, y_true, y_pred, conf, names, ref_support, out_dir):
    valid = np.ones(len(y_true), bool)          # already filtered to classes with ref
    acc = (y_true == y_pred).mean()
    classes = sorted(set(y_true.tolist()))
    per_class = {}
    for c in classes:
        m = y_true == c
        per_class[c] = ((y_pred[m] == c).mean(), int(m.sum()))
    macro = np.mean([per_class[c][0] for c in classes])

    lines = [f"=== METHOD: {name} ===",
             f"overall top-1 accuracy : {acc:.4f}  (n={len(y_true)})",
             f"macro per-class recall : {macro:.4f}  (classes={len(classes)})",
             "",
             f"{'id':>3} {'class':<28} {'ref':>5} {'qry':>5} {'acc':>7}",
             "-" * 52]
    for c in classes:
        a, sup = per_class[c]
        lines.append(f"{c:>3} {names[c][:28]:<28} {ref_support.get(c,0):>5} {sup:>5} {a:>7.3f}")

    # top confusions
    conf_pairs = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf_pairs[(t, p)] += 1
    lines += ["", "top confusions (true -> pred : count):"]
    for (t, p), n in conf_pairs.most_common(15):
        lines.append(f"  {names[t][:24]:<24} -> {names[p][:24]:<24} : {n}")

    # rejection sweep (single-proto/knn conf already in [0,1]-ish)
    lines += ["", "rejection sweep (min-confidence -> coverage, accuracy-on-kept):"]
    for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        keep = conf >= thr
        if keep.sum() == 0:
            lines.append(f"  thr={thr:.2f}  coverage=0.000  acc=  n/a"); continue
        cov = keep.mean()
        a = (y_true[keep] == y_pred[keep]).mean()
        lines.append(f"  thr={thr:.2f}  coverage={cov:.3f}  acc={a:.4f}")

    report = "\n".join(lines)
    log("\n" + report + "\n")
    (Path(out_dir) / f"report_{name}.txt").write_text(report)
    return {"method": name, "acc": acc, "macro_recall": macro, "n": len(y_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/retrocausal-train/Documents/RPN+DinoV2/datasets/gas_valve_2view")
    ap.add_argument("--classes_txt", default=None, help="default: <data_dir>/classes.txt")
    ap.add_argument("--out_dir", default="/home/retrocausal-train/Documents/RPN+DinoV2/phase2_results")
    ap.add_argument("--ref_frames", type=int, default=300)
    ap.add_argument("--query_frames", type=int, default=1000, help="-1 = all remaining")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampling", choices=["even", "random"], default="even",
                    help="even = uniform stride across sorted frames (recommended for video); "
                         "random = shuffle by seed")
    ap.add_argument("--min_frames_per_class", type=int, default=20,
                    help="even sampling: guarantee at least this many reference frames "
                         "cover each class (tops up the even-spread base for rare classes)")
    ap.add_argument("--samples_per_class", type=int, default=0,
                    help="if >0, cap the reference set to EXACTLY this many bbox crops per "
                         "class (evenly spread over time); balances common vs rare classes. "
                         "Frame floor is raised to this value so enough crops are available.")
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["centroid", "knn", "cluster"])
    ap.add_argument("--ref_all", action="store_true",
                    help="use EVERY non-query frame as reference (max examples per class). "
                         "Query is picked first (even stride of --query_frames), reference is "
                         "the entire remaining pool. Slower inference, best accuracy.")
    args = ap.parse_args()

    if (args.samples_per_class > 0 and not args.ref_all
            and args.min_frames_per_class < args.samples_per_class):
        # need >= N frames covering each class to be able to keep N crops per class
        # (ref_all already loads every frame, so no frame-floor top-up is needed)
        args.min_frames_per_class = args.samples_per_class

    data_dir = args.data_dir
    classes_txt = args.classes_txt or str(Path(data_dir) / "classes.txt")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    names = load_classes(classes_txt)
    log(f"classes: {len(names)}")

    stems = list_frames(data_dir)          # sorted by filename == chronological
    log(f"frames with labels: {len(stems)}")

    if args.ref_all:
        # invert the split: query first (even stride), reference = ALL remaining frames
        n = len(stems)
        q = min(args.query_frames if args.query_frames != -1 else n, n)
        pick = np.linspace(0, n - 1, q).round().astype(int)
        query_idx = sorted(set(pick.tolist()))
        query_set = set(query_idx)
        ref_idx = [i for i in range(n) if i not in query_set]
        query_stems = [stems[i] for i in query_idx]
        ref_stems = [stems[i] for i in ref_idx]
        log(f"  ref_all: query {len(query_stems)} frames -> reference = all {len(ref_stems)} "
            f"remaining frames (max examples/class)")
    elif args.sampling == "even":
        # reference = uniform stride across the whole timeline (both views, all phases)
        n = len(stems)
        base = set(int(x) for x in np.linspace(0, n - 1, args.ref_frames).round().astype(int))

        # guarantee >= min_frames_per_class reference frames per class: top up rare
        # classes with evenly-spread frames that contain them (rarest first)
        per_frame, by_class = frame_class_index(data_dir, stems)
        floor = args.min_frames_per_class
        short = []
        for c in sorted(by_class, key=lambda c: len(by_class[c])):
            covered = sum(1 for i in base if c in per_frame[i])
            if covered >= floor:
                continue
            cands = [i for i in by_class[c] if i not in base]
            need = floor - covered
            if len(cands) <= need:
                base.update(cands)          # take all available
            else:
                pick = np.linspace(0, len(cands) - 1, need).round().astype(int)
                base.update(cands[j] for j in sorted(set(pick.tolist())))
            final = sum(1 for i in base if c in per_frame[i])
            if final < floor:
                short.append((names[c], final))
        if short:
            log(f"  classes with < {floor} frames (fewer exist in dataset): {short}")
        log(f"  even base {args.ref_frames} -> {len(base)} frames after >= {floor}/class top-up")

        ref_idx = sorted(base)
        ref_set = set(ref_idx)
        ref_stems = [stems[i] for i in ref_idx]
        rest_idx = [i for i in range(n) if i not in ref_set]
        if args.query_frames == -1:
            query_idx = rest_idx
        else:                               # query = even stride over the remaining frames
            q = min(args.query_frames, len(rest_idx))
            pick = np.linspace(0, len(rest_idx) - 1, q).round().astype(int)
            query_idx = sorted(set(rest_idx[j] for j in pick))
        query_stems = [stems[i] for i in query_idx]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(stems)
        ref_stems = stems[:args.ref_frames]
        rest = stems[args.ref_frames:]
        q = len(rest) if args.query_frames == -1 else min(args.query_frames, len(rest))
        query_stems = rest[:q]

    q = len(query_stems)
    log(f"sampling={args.sampling}  reference frames: {len(ref_stems)}  |  query frames: {q}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-base"
    log(f"loading {model_name} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)

    ref_tag = "all" if args.ref_all else str(args.ref_frames)
    cache = Path(out_dir) / (f"cache_{args.sampling}_seed{args.seed}_ref{ref_tag}"
                             f"_min{args.min_frames_per_class}_q{q}.npz")
    if cache.exists():
        log(f"loading cached embeddings: {cache.name}")
        d = np.load(cache)
        Xr, yr, Xq, yq, aq = d["Xr"], d["yr"], d["Xq"], d["yq"], d["aq"]
    else:
        log("cropping + embedding REFERENCE ...")
        rc, ry, ra, rsk = crop_frames(data_dir, ref_stems, args.min_box_px)
        log(f"  ref crops: {len(rc)} (skipped {rsk} tiny)")
        Xr = embed(rc, processor, model, device, args.batch); yr = ry
        log("cropping + embedding QUERY ...")
        qc, qy, qa, qsk = crop_frames(data_dir, query_stems, args.min_box_px)
        log(f"  query crops: {len(qc)} (skipped {qsk} tiny)")
        Xq = embed(qc, processor, model, device, args.batch); yq = qy; aq = qa
        np.savez_compressed(cache, Xr=Xr, yr=yr, Xq=Xq, yq=yq, aq=aq)
        log(f"cached -> {cache.name}")

    if args.samples_per_class > 0:
        before = Counter(yr.tolist())
        Xr, yr = cap_per_class(Xr, yr, args.samples_per_class, seed=args.seed)
        after = Counter(yr.tolist())
        short = {names[c]: after[c] for c in sorted(after) if after[c] < args.samples_per_class}
        log(f"capped reference to {args.samples_per_class} crops/class: "
            f"{sum(before.values())} -> {len(yr)} exemplars across {len(after)} classes")
        if short:
            log(f"  classes with < {args.samples_per_class} available (kept all): {short}")

    ref_support = Counter(yr.tolist())
    ref_classes = set(yr.tolist())
    # query instances whose class has NO reference exemplars can never be right
    coverable = np.array([c in ref_classes for c in yq])
    n_uncoverable = int((~coverable).sum())
    missing = sorted(set(yq[~coverable].tolist()))
    log(f"\nref covers {len(ref_classes)}/{len(names)} classes")
    if missing:
        log(f"query has {n_uncoverable} instances in {len(missing)} classes absent from reference "
            f"(excluded from scoring): {[names[c] for c in missing]}")
    Xq_e, yq_e, aq_e = Xq[coverable], yq[coverable], aq[coverable]

    summary = []
    if "centroid" in args.methods:
        protos, pl = build_centroids(Xr, yr)
        pred, conf = predict_nearest_proto(Xq_e, protos, pl)
        summary.append(evaluate("centroid", yq_e, pred, conf, names, ref_support, out_dir))
    if "cluster" in args.methods:
        protos, pl = build_cluster_prototypes(Xr, yr)
        log(f"cluster prototypes: {len(pl)} across {len(set(pl.tolist()))} classes")
        pred, conf = predict_nearest_proto(Xq_e, protos, pl)
        summary.append(evaluate("cluster", yq_e, pred, conf, names, ref_support, out_dir))
    if "knn" in args.methods:
        pred, conf = predict_knn(Xq_e, Xr, yr, k=args.knn_k)
        summary.append(evaluate("knn", yq_e, pred, conf, names, ref_support, out_dir))

    log("\n==== SUMMARY ====")
    log(f"{'method':<10} {'acc':>8} {'macro_recall':>14} {'n':>7}")
    for s in summary:
        log(f"{s['method']:<10} {s['acc']:>8.4f} {s['macro_recall']:>14.4f} {s['n']:>7}")
    log(f"\nreports written to {out_dir}/report_<method>.txt")


if __name__ == "__main__":
    main()
