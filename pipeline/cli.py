"""Command-line entry point.

    python -m pipeline.cli <step> --config configs/gas_valve.yaml

steps: prepare | train | reference | infer | evaluate | visualize | visualize-masks | all

The configured `phase1.device` is pinned via CUDA_VISIBLE_DEVICES *before* torch is
imported, and every stage then uses local index 0 — this avoids the device-ordinal
clash that Ultralytics' CUDA_VISIBLE_DEVICES remapping otherwise causes when training
and embedding run in the same process.
"""
from __future__ import annotations

import argparse
import os

from .config import Config  # yaml only, no torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "train", "reference", "infer",
                                     "evaluate", "visualize", "visualize-masks", "all"])
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--weights", default=None, help="detector weights for `infer`")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the Phase-2 reference cache")
    ap.add_argument("--weight", type=float, default=None,
                    help="`evaluate` only: override the YOLO/DINOv2 fusion weight w in "
                         "(w)*yolo + (1-w)*dino for a phase2.fuse_multiclass run "
                         "(default: pure YOLO, w=1.0). The report also always includes "
                         "a w=0..1 sweep table regardless.")
    args = ap.parse_args(argv)

    cfg = Config(args.config)

    # Pin the GPU before any torch/ultralytics import, then use local index 0.
    dev = str(cfg.phase1.device)
    if dev.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
        cfg.phase1.device = "0"

    # Use locally-cached HF models (avoids network / stale-token 401 on long runs).
    if getattr(cfg.phase2, "hf_offline", False):
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            os.environ.setdefault(k, "1")

    from . import pipeline as P  # imports torch/ultralytics — after GPU is pinned

    print(f"config: {cfg.path}\nroot: {cfg.root}\nwork_dir: {cfg.work}\n"
          f"classes: {len(cfg.classes)}   gpu(CUDA_VISIBLE_DEVICES)={os.environ.get('CUDA_VISIBLE_DEVICES','?')}\n")

    steps = {
        "prepare": lambda: P.prepare(cfg),
        "train": lambda: P.train(cfg),
        "reference": lambda: P.reference(cfg, rebuild=args.rebuild),
        "infer": lambda: P.infer(cfg, weights=args.weights),
        "evaluate": lambda: P.evaluate(cfg, weight=args.weight),
        "visualize": lambda: P.visualize(cfg),
        "visualize-masks": lambda: P.visualize_masks(cfg),
        "all": lambda: P.run_all(cfg),
    }
    steps[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
