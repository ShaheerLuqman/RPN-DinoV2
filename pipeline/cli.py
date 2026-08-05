"""Command-line entry point.

    python -m pipeline.cli <step> --config configs/doosan_swivel.yaml

steps: prepare | train | infer | evaluate | visualize | all

The configured `phase1.device` is pinned via CUDA_VISIBLE_DEVICES *before* torch is
imported, and every stage then uses local index 0 — this avoids the device-ordinal
clash that Ultralytics' CUDA_VISIBLE_DEVICES remapping otherwise causes.
"""
from __future__ import annotations

import argparse
import os

from .config import Config  # yaml only, no torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "train", "infer",
                                     "evaluate", "visualize", "all"])
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--weights", default=None, help="detector weights for `infer`")
    args = ap.parse_args(argv)

    cfg = Config(args.config)

    # Pin the GPU before any torch/ultralytics import, then use local index 0.
    dev = str(cfg.phase1.device)
    if dev.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
        cfg.phase1.device = "0"

    from . import pipeline as P  # imports torch/ultralytics — after GPU is pinned

    print(f"config: {cfg.path}\nroot: {cfg.root}\nwork_dir: {cfg.work}\n"
          f"classes: {len(cfg.classes)}   gpu(CUDA_VISIBLE_DEVICES)={os.environ.get('CUDA_VISIBLE_DEVICES','?')}\n")

    steps = {
        "prepare": lambda: P.prepare(cfg),
        "train": lambda: P.train(cfg),
        "infer": lambda: P.infer(cfg, weights=args.weights),
        "evaluate": lambda: P.evaluate(cfg),
        "visualize": lambda: P.visualize(cfg),
        "all": lambda: P.run_all(cfg),
    }
    steps[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
