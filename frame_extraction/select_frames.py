#!/usr/bin/env python3
"""Pick a diverse, non-redundant frame subset for annotation — from raw video or an
already-extracted frame folder — with no labels and no trained model required.

Design doc: docs/frame_selection.md (every choice below was locked there, not picked
ad hoc here — see that doc for the "why").

Pipeline, per source (a video file, or a frame-folder group — see below):
    1. read every native frame
    2. quality pre-filter: drop blurry / over-under-exposed / lens-occluded frames
       (classical CV, no model — Laplacian-variance blur, mean-brightness exposure,
       low full-frame std as a lens-occlusion heuristic)
    3. embed survivors with a frozen DINOv2 backbone (CLS token, L2-normalized so
       cosine distance == dot product) — batched, cached to disk under this run's
       own folder (<run-dir>/cache) so a re-run with a different budget doesn't
       re-pay the embedding cost
    4. adaptive pre-bucketing: walk frames in temporal order, keep one representative
       whenever cosine distance from the last-kept representative exceeds --bucket-delta.
       Bucket size self-adjusts to how fast the content is actually changing, instead of
       a fixed wall-clock window — a fixed window mis-sizes across cycles that run at
       different speeds (a fast/efficient cycle packs real events into less wall-clock
       time than a slow one with delays).

Then, pooling every source's representatives together (global scope — catches
redundancy *across* videos/sources, not just within one):
    5. k-center greedy (farthest-point sampling, cosine distance) over the pooled
       representatives until --keep-fraction of the total post-filter candidate count
       is selected
    6. write a manifest (JSON) of the selected frame identifiers, in pick order
    7. stage the selected frames as actual files in a folder (skip with --no-stage):
       symlinked from the source image when one exists on disk (frame-folder input —
       falls back to a hardlink, then a plain copy, on platforms that disallow
       symlinks; same fallback pipeline.dataset._link_or_copy uses), written out from
       the decoded frame otherwise (video input). Sibling YOLO labels and classes.txt
       are symlinked alongside when the input was an already-labeled frame folder.

Input: pass exactly one of --images-dir (a frame folder) or --videos-dir (raw videos).
A frame folder can be either a prepare_dataset.py-staged flat sequence (000001.jpg,
...) or per-video-named (<video>_frame_000205.jpg) — the source-video naming pattern
is auto-detected; a flat sequence already preserves source-video + frame order via its
sequential renumbering (see prepare_dataset.py), it just doesn't carry the name.

Usage:
    # config-driven (see configs/*_frame_selection.yaml) — same --config convention
    # as pipeline/cli.py; copy the YAML for a new run rather than mixing in flags:
    python frame_extraction/select_frames.py --config configs/doosan_tackweld_frame_selection.yaml

    # or pure CLI args, e.g. an already-extracted, already-processed frame folder:
    python frame_extraction/select_frames.py \\
        --images-dir datasets/doosan_tackweld_processed --image-ext jpg \\
        --keep-fraction 0.05 \\
        --out runs/frame_selection/doosan_tackweld_selected.json

    # raw video input instead:
    python frame_extraction/select_frames.py \\
        --videos-dir datasets/marmon_station_1/videos --video-ext mkv \\
        --keep-fraction 0.05
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dataset import list_frames, _link_or_copy  # noqa: E402

DEFAULT_MODEL = "facebook/dinov2-base"
# same "<name>_frame_000205" / "<name>.mp4_1000" / "<name>.mkv_1000" convention
# prepare_dataset.py stages from — kept identical so frame-folder grouping agrees
# with how that script recovers source-video order.
FRAME_RE = re.compile(r"^(.*?)(?:_frame_(\d+)|\.(?:mp4|mkv)_(\d+))$")


# ---------------------------------------------------------------------------
# Frame sources — yield (frame_id, BGR ndarray, source_path) in temporal order.
# source_path is the on-disk image file for frame-folder input (so staging can
# symlink it directly) or None for video input (no per-frame file exists until
# staging writes one).
# ---------------------------------------------------------------------------

def _group_frame_folder(stems: list[str]) -> dict[str, list[str]]:
    """Group stems by source video if the per-video naming convention is
    recognizable; otherwise one group holding every stem in (already temporally
    sorted) order — the prepare_dataset.py flat-sequence case."""
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    matched = 0
    for i, stem in enumerate(stems):
        m = FRAME_RE.match(stem)
        if m:
            matched += 1
            video = m.group(1)
            frame_no = int(m.group(2) or m.group(3))
            groups[video].append((frame_no, stem))
        else:
            groups["_all"].append((i, stem))
    if matched == 0:
        return {"_all": list(stems)}
    return {video: [s for _, s in sorted(items)] for video, items in groups.items()}


def _iter_folder_group(images_dir: Path, ext: str, stems: list[str]):
    for stem in stems:
        path = images_dir / f"{stem}.{ext}"
        img = cv2.imread(str(path))
        if img is not None:
            yield stem, img, path


def _iter_video(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield f"frame_{idx:06d}", frame, None
        idx += 1
    cap.release()


def iter_sources(args) -> dict[str, object]:
    """-> {group_name: generator of (frame_id, BGR ndarray, source_path)}, one group
    per video / per recognized frame-folder group. Lazy — nothing is decoded until
    the caller actually iterates a group, so memory stays bounded by however much
    of one group the caller chooses to hold at once (see `process_group`), not by
    total dataset size. (An earlier version materialized every frame into a list
    here; fine at ~4k frames, ~24GB+ RSS and 30GB+ swap at ~23k — see git history.)"""
    if args.images_dir:
        images_dir = Path(args.images_dir)
        images_dir = images_dir if images_dir.is_absolute() else REPO_ROOT / images_dir
        stems = list_frames_any(images_dir, args.image_ext)
        if not stems:
            raise FileNotFoundError(f"no *.{args.image_ext} frames under {images_dir}")
        groups = _group_frame_folder(stems)
        return {name: _iter_folder_group(images_dir, args.image_ext, s)
                for name, s in groups.items()}
    else:
        videos_dir = Path(args.videos_dir)
        videos_dir = videos_dir if videos_dir.is_absolute() else REPO_ROOT / videos_dir
        videos = sorted(videos_dir.glob(f"*.{args.video_ext}"))
        if not videos:
            raise FileNotFoundError(f"no *.{args.video_ext} files under {videos_dir}")
        return {v.stem: _iter_video(v) for v in videos}


def list_frames_any(images_dir: Path, ext: str) -> list[str]:
    """Like pipeline.dataset.list_frames, but doesn't require a sibling .txt label
    (frame-folder input here may be unlabeled raw-extracted frames, not a
    prepare_dataset.py-staged labeled set) — falls back to every image present if
    none have labels, else defers to list_frames so an already-labeled folder (e.g.
    datasets/doosan_tackweld_processed) is handled identically either way."""
    labeled = list_frames(images_dir, ext)
    if labeled:
        return labeled
    return [p.stem for p in sorted(images_dir.glob(f"*.{ext}"))]


# ---------------------------------------------------------------------------
# Quality pre-filter — blur, exposure, occlusion (classical CV, no model)
# ---------------------------------------------------------------------------

def quality_ok(frame_bgr: np.ndarray, blur_thresh: float, exposure_range: tuple[float, float],
              occlusion_std_thresh: float) -> tuple[bool, list[str]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())

    reasons = []
    if blur_score < blur_thresh:
        reasons.append("blur")
    if not (exposure_range[0] <= mean_brightness <= exposure_range[1]):
        reasons.append("exposure")
    if std_brightness < occlusion_std_thresh:
        reasons.append("occlusion")
    return not reasons, reasons


# ---------------------------------------------------------------------------
# DINOv2 embedding, batched, frozen (no fine-tuning)
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def embed_batch(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()  # CLS token
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        return (cls / np.clip(norm, 1e-8, None)).astype(np.float32)  # cosine == dot


def _save_cache(cache_path: Path, cache_map: dict[str, np.ndarray]) -> None:
    all_ids = list(cache_map.keys())
    all_emb = np.stack([cache_map[i] for i in all_ids]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(all_ids, dtype=object), embeddings=all_emb)
    tmp.replace(cache_path)  # atomic-ish swap so a crash mid-write can't corrupt the cache


class CachingEmbedder:
    """Wraps `Embedder` with an on-disk cache keyed by frame id, loaded once and
    shared across every group in the run (rather than reloaded/rewritten per
    group). Only ever holds the cache's *embeddings* in memory (a few KB per
    frame — ~70MB for 23k frames), never raw pixels — the caller is responsible
    for only passing one batch of raw frames at a time (see `process_group`).
    Persists to disk every `save_every_frames` newly-embedded frames, not just at
    the end, so an interrupted run only loses a few batches of progress."""

    def __init__(self, embedder: Embedder, cache_path: Path, save_every_frames: int = 320):
        self.embedder = embedder
        self.cache_path = cache_path
        self.save_every_frames = save_every_frames
        self.cache_map: dict[str, np.ndarray] = {}
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.cache_map = {i: e for i, e in zip(data["ids"], data["embeddings"])}
        self.n_cache_hits = 0
        self.n_newly_embedded = 0
        self._unsaved = 0

    def embed_batch(self, ids: list[str], imgs: list[np.ndarray]) -> np.ndarray:
        missing = [i for i, fid in enumerate(ids) if fid not in self.cache_map]
        self.n_cache_hits += len(ids) - len(missing)
        if missing:
            embs = self.embedder.embed_batch([imgs[i] for i in missing])
            for j, i in enumerate(missing):
                self.cache_map[ids[i]] = embs[j]
            self.n_newly_embedded += len(missing)
            self._unsaved += len(missing)
            if self._unsaved >= self.save_every_frames:
                self.flush()
                print(f"      ...{self.n_newly_embedded} newly embedded so far, cache saved", flush=True)
        return np.stack([self.cache_map[fid] for fid in ids]).astype(np.float32)

    def flush(self) -> None:
        if self._unsaved:
            _save_cache(self.cache_path, self.cache_map)
            self._unsaved = 0


def process_group(gname: str, frame_source, cache: CachingEmbedder, args) -> dict:
    """Streams `frame_source` (one group's frames, in temporal order) through
    quality-filter -> embed -> adaptive-bucket, holding at most one --batch-size
    chunk of raw frames in memory at a time — not the whole group. Bucketing
    itself only ever needs the last-kept embedding, so it composes with
    streaming batches without changing the result (same sequential comparison
    as processing the group in one shot, just chunked)."""
    rep_ids, rep_paths, rep_frames, rep_embs = [], [], [], []
    last_emb = None
    batch_ids, batch_imgs, batch_paths = [], [], []
    n_seen = n_kept = 0

    def flush_batch():
        nonlocal last_emb
        if not batch_ids:
            return
        embs = cache.embed_batch(batch_ids, batch_imgs)
        for j in range(len(batch_ids)):
            e = embs[j]
            if last_emb is None or (1.0 - float(np.dot(e, last_emb))) > args.bucket_delta:
                rep_ids.append(batch_ids[j])
                rep_paths.append(batch_paths[j])
                # only hold the decoded frame in memory when there's no source
                # file to symlink from later (video input) — frame-folder input
                # re-reads from disk at staging time instead
                rep_frames.append(None if batch_paths[j] is not None else batch_imgs[j])
                rep_embs.append(e)
                last_emb = e
        batch_ids.clear()
        batch_imgs.clear()
        batch_paths.clear()

    for frame_id, img, src_path in frame_source:
        n_seen += 1
        ok, _reasons = quality_ok(img, args.blur_thresh, tuple(args.exposure_range),
                                  args.occlusion_std_thresh)
        if not ok:
            continue
        n_kept += 1
        batch_ids.append(f"{gname}::{frame_id}")
        batch_imgs.append(img)
        batch_paths.append(src_path)
        if len(batch_ids) >= args.batch_size:
            flush_batch()
    flush_batch()

    return {"ids": rep_ids, "paths": rep_paths, "frames": rep_frames,
            "embs": np.stack(rep_embs).astype(np.float32) if rep_embs else np.zeros((0, 0), np.float32),
            "n_seen": n_seen, "n_kept": n_kept}


# ---------------------------------------------------------------------------
# k-center greedy (farthest-point sampling), cosine distance
# ---------------------------------------------------------------------------

def k_center_greedy(embeddings: np.ndarray, k: int, seed_idx: int = 0) -> list[int]:
    n = len(embeddings)
    if k >= n:
        return list(range(n))
    selected = [seed_idx]
    min_dist = 1.0 - embeddings @ embeddings[seed_idx]
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - embeddings @ embeddings[nxt])
    return selected


# ---------------------------------------------------------------------------
# Staging — materialize the selected subset as files in a folder: symlinked from
# the source when one exists on disk (frame-folder input, falling back to
# hardlink/copy on platforms that don't allow symlinks — see
# pipeline.dataset._link_or_copy), written out from the decoded frame otherwise
# (video input, where no per-frame file exists until now).
# ---------------------------------------------------------------------------

def stage_selected(stage_dir: Path, ext: str, entries: list[tuple[str, str, Path | None, np.ndarray | None]],
                   images_dir: Path | None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for out_name, _pick_id, src_path, raw_frame in entries:
        img_dst = stage_dir / f"{out_name}.{ext}"
        if src_path is not None:
            _link_or_copy(src_path.resolve(), img_dst)
            lbl_src = src_path.with_suffix(".txt")
            if lbl_src.exists():
                _link_or_copy(lbl_src.resolve(), stage_dir / f"{out_name}.txt")
        else:
            cv2.imwrite(str(img_dst), raw_frame)

    if images_dir is not None:
        classes_src = images_dir / "classes.txt"
        if classes_src.exists():
            _link_or_copy(classes_src.resolve(), stage_dir / "classes.txt")


def load_config(args: argparse.Namespace, config_path: Path) -> None:
    """Overwrites every field on `args` from a YAML config — `--config` is the
    whole interface when given, same convention pipeline/cli.py's configs/*.yaml
    use (edit/copy the YAML for a new run rather than mixing in other flags; see
    README_pipeline.md's sweep-via-sed example). `project.root: auto` resolves
    the same way pipeline/config.py's does: the repo dir containing configs/."""
    import yaml
    raw = yaml.safe_load(config_path.read_text()) or {}

    root = ((raw.get("project") or {}).get("root")) or "auto"
    proj_root = config_path.parent.parent if root == "auto" else Path(root).expanduser().resolve()

    def resolve(p) -> str:
        p = Path(p).expanduser()
        return str(p if p.is_absolute() else proj_root / p)

    def resolve_auto(section: dict, key: str) -> str | None:
        v = section.get(key, "auto")
        return None if v in (None, "auto") else resolve(v)

    source = raw.get("source") or {}
    if "images_dir" in source:
        args.images_dir, args.videos_dir = resolve(source["images_dir"]), None
    elif "videos_dir" in source:
        args.videos_dir, args.images_dir = resolve(source["videos_dir"]), None
    else:
        raise ValueError(f"{config_path}: `source:` must set images_dir or videos_dir")
    args.image_ext = source.get("image_ext", args.image_ext)
    args.video_ext = source.get("video_ext", args.video_ext)

    sel = raw.get("selection") or {}
    args.keep_fraction = sel.get("keep_fraction", args.keep_fraction)
    args.bucket_delta = sel.get("bucket_delta", args.bucket_delta)
    args.seed_index = sel.get("seed_index", args.seed_index)

    qf = raw.get("quality_filter") or {}
    args.blur_thresh = qf.get("blur_thresh", args.blur_thresh)
    args.exposure_range = qf.get("exposure_range", args.exposure_range)
    args.occlusion_std_thresh = qf.get("occlusion_std_thresh", args.occlusion_std_thresh)

    emb = raw.get("embedding") or {}
    args.model = emb.get("model", args.model)
    args.device = emb.get("device", args.device)
    args.batch_size = emb.get("batch_size", args.batch_size)

    out = raw.get("output") or {}
    args.run_dir = resolve_auto(out, "run_dir")
    args.cache_dir = resolve_auto(out, "cache_dir")
    args.out = resolve_auto(out, "out")
    args.stage_dir = resolve_auto(out, "stage_dir")
    args.no_stage = not out.get("stage", True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config (see configs/*_frame_selection.yaml) — when given, every other "
                         "flag below is ignored, same convention as pipeline/cli.py's --config")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--images-dir", default=None, help="a frame folder (labeled or not)")
    src.add_argument("--videos-dir", default=None, help="a folder of raw videos")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--video-ext", default="mkv")

    ap.add_argument("--keep-fraction", type=float, default=0.05,
                    help="fraction of post-quality-filter candidate frames to select globally (default: 0.05 = 5%%)")
    ap.add_argument("--bucket-delta", type=float, default=0.10,
                    help="cosine-distance threshold for adaptive pre-bucketing (default: 0.10) — "
                         "tuning constant, see docs/frame_selection.md")
    ap.add_argument("--blur-thresh", type=float, default=60.0,
                    help="Laplacian-variance floor; frames below this are dropped as blurry")
    ap.add_argument("--exposure-range", type=float, nargs=2, default=[15.0, 240.0], metavar=("LOW", "HIGH"),
                    help="acceptable mean-brightness range (0-255); frames outside are dropped")
    ap.add_argument("--occlusion-std-thresh", type=float, default=5.0,
                    help="brightness-std floor; a near-uniform frame below this is dropped as lens-occluded")

    ap.add_argument("--model", default=DEFAULT_MODEL, help="frozen HF backbone used for frame embeddings")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed-index", type=int, default=0, help="k-center greedy starting index into the pooled set")

    ap.add_argument("--run-dir", default=None,
                    help="folder everything for this run lands under (manifest, staged frames, "
                         "embedding cache) — default: runs/frame_selection/<source-name>")
    ap.add_argument("--cache-dir", default=None, help="default: <run-dir>/cache")
    ap.add_argument("--out", default=None, help="default: <run-dir>/selected.json")
    ap.add_argument("--stage-dir", default=None,
                    help="folder to materialize the selected frames into (symlinked from the source "
                         "when one exists on disk, else written from the decoded frame) — "
                         "default: <run-dir>/selected")
    ap.add_argument("--no-stage", action="store_true", help="skip materializing files, manifest only")
    args = ap.parse_args()

    if args.config:
        config_path = Path(args.config)
        config_path = config_path if config_path.is_absolute() else REPO_ROOT / config_path
        load_config(args, config_path)
    elif not args.images_dir and not args.videos_dir:
        ap.error("one of --images-dir, --videos-dir, or --config is required")

    source_path = Path(args.images_dir or args.videos_dir)
    # a videos_dir is conventionally a "videos" subfolder nested under the dataset's
    # own name (e.g. datasets/marmon_station_1/videos) — use the parent name in that
    # case so runs land under a descriptive, collision-free run dir instead of
    # everyone's videos_dir naming the same generic "videos" run.
    source_name = source_path.parent.name if (args.videos_dir and source_path.name == "videos") else source_path.name
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "runs" / "frame_selection" / source_name
    cache_dir = Path(args.cache_dir) if args.cache_dir else run_dir / "cache"
    out_path = Path(args.out) if args.out else run_dir / "selected.json"
    stage_dir = Path(args.stage_dir) if args.stage_dir else run_dir / "selected"
    cache_path = cache_dir / f"embeddings_{args.model.replace('/', '_')}.npz"

    print(f"Loading {args.model} (frozen) on {args.device} ...")
    embedder = Embedder(args.model, args.device)

    print(f"Scanning source ({'frame folder' if args.images_dir else 'videos'}: "
          f"{args.images_dir or args.videos_dir}) ...")
    groups = iter_sources(args)
    print(f"  {len(groups)} group(s) — streaming (quality-filter -> embed -> bucket "
          f"one --batch-size chunk at a time, not the whole dataset at once)")

    cache = CachingEmbedder(embedder, cache_path, save_every_frames=10 * args.batch_size)

    pooled_ids: list[str] = []       # "group::frame_id"
    pooled_paths: list[Path | None] = []   # source image path (frame-folder input) or None (video input)
    pooled_frames: list[np.ndarray | None] = []  # decoded frame, kept only when pooled_paths[i] is None
    pooled_emb_list: list[np.ndarray] = []
    n_total = 0
    n_quality_dropped = 0
    n_candidates = 0

    for gname, frame_source in groups.items():
        result = process_group(gname, frame_source, cache, args)
        n_total += result["n_seen"]
        n_quality_dropped += result["n_seen"] - result["n_kept"]
        if result["n_kept"] == 0:
            continue
        n_candidates += result["n_kept"]

        print(f"  [{gname}] {result['n_kept']}/{result['n_seen']} pass quality filter, "
              f"{len(result['ids'])}/{result['n_kept']} survive adaptive pre-bucketing (delta={args.bucket_delta})")
        pooled_ids.extend(result["ids"])
        pooled_paths.extend(result["paths"])
        pooled_frames.extend(result["frames"])
        pooled_emb_list.append(result["embs"])

    cache.flush()
    print(f"  embeddings: {cache.n_cache_hits} reused from cache, {cache.n_newly_embedded} newly computed")

    if not pooled_ids:
        raise RuntimeError("no candidate frames survived the quality filter")

    pooled_emb = np.concatenate(pooled_emb_list, axis=0)
    budget = max(1, round(args.keep_fraction * n_candidates))
    budget = min(budget, len(pooled_ids))

    print(f"\nPooled {len(pooled_ids)} representative frames (from {n_candidates} quality-passed "
          f"candidates, {n_total} total). Selecting {budget} ({args.keep_fraction:.1%} of candidates) via k-center ...")
    t0 = time.time()
    picked = k_center_greedy(pooled_emb, budget, seed_idx=args.seed_index)
    print(f"  done in {time.time() - t0:.1f}s")

    manifest = {
        "source": args.images_dir or args.videos_dir,
        "model": args.model,
        "params": {
            "keep_fraction": args.keep_fraction, "bucket_delta": args.bucket_delta,
            "blur_thresh": args.blur_thresh, "exposure_range": args.exposure_range,
            "occlusion_std_thresh": args.occlusion_std_thresh, "seed_index": args.seed_index,
        },
        "counts": {
            "total_frames": n_total, "quality_dropped": n_quality_dropped,
            "candidates": n_candidates, "bucket_representatives": len(pooled_ids),
            "selected": len(picked),
        },
        "selected": [{"id": pooled_ids[i], "pick_order": order} for order, i in enumerate(picked)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{n_total} total frames -> {n_quality_dropped} dropped (quality) -> {n_candidates} candidates "
          f"-> {len(pooled_ids)} after pre-bucketing -> {len(picked)} selected ({len(picked) / n_total:.1%} of total)")
    print(f"-> {out_path}")

    if not args.no_stage:
        # single group ("_all", the flat prepare_dataset.py-style case) keeps the
        # original stem as the filename; multi-group (per-video) pools disambiguate
        # with a "<group>__<frame_id>" name since frame_id alone can collide across groups
        single_group = len(groups) == 1 and "_all" in groups
        entries = []
        for order, i in enumerate(picked):
            gname, frame_id = pooled_ids[i].split("::", 1)
            out_name = frame_id if single_group else f"{gname}__{frame_id}"
            entries.append((out_name, pooled_ids[i], pooled_paths[i], pooled_frames[i]))
        images_dir = None
        if args.images_dir:
            p = Path(args.images_dir)
            images_dir = p if p.is_absolute() else REPO_ROOT / p
        stage_selected(stage_dir, args.image_ext if args.images_dir else "jpg", entries, images_dir)
        print(f"-> {stage_dir}  ({len(entries)} frames staged" +
              (" + labels + classes.txt" if args.images_dir else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
