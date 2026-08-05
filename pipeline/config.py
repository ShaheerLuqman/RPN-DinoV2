"""Load and resolve the YAML pipeline config.

Every path in the config is resolved relative to `project.root` (unless absolute),
so a config is portable across machines by editing that one field (or setting it to
"auto" to use the repo directory that contains the config's parent).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


class Config:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.raw = yaml.safe_load(self.path.read_text())

        root = (self.raw.get("project", {}) or {}).get("root") or "auto"
        if root == "auto":
            # repo root = parent of the config file's parent (…/repo/configs/x.yaml)
            self.root = self.path.parent.parent
        else:
            self.root = Path(root).expanduser().resolve()

        for section in ("data", "phase1", "infer", "eval", "visualize"):
            setattr(self, section, _ns(self.raw.get(section, {}) or {}))

        # `{frames}` in work_dir is auto-filled so each run is self-labeling
        # (e.g. runs/4079_{frames} -> runs/4079_300).
        frames = str((self.raw.get("data") or {}).get("annotation_frames", ""))
        wd = str(self.raw["project"]["work_dir"]).replace("{frames}", frames)
        self.work = self.resolve(wd)
        self.work.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.resolve(self.data.images_dir)
        self.classes = self._load_classes()

    def resolve(self, p) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def _load_classes(self) -> list[str]:
        f = self.resolve(self.data.classes_file)
        return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]

    # convenient artifact paths under work_dir
    @property
    def dataset_dir(self) -> Path:
        return self.work / "dataset"

    @property
    def detector_dir(self) -> Path:
        return self.work / "detector"

    @property
    def best_weights(self) -> Path:
        return self.detector_dir / "train" / "weights" / "best.pt"

    @property
    def predictions_json(self) -> Path:
        return self.work / "predictions.json"
