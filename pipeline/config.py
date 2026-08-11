"""Load and resolve the YAML pipeline config.

Every path in the config is resolved relative to `project.root` (unless absolute),
so a config is portable across machines by editing that one field (or setting it to
"auto" to use the repo directory that contains the config's parent).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

# canonical pipeline order; `project.steps` (if given) only filters this down —
# it never reorders `run_all`
ALL_STEPS = ("prepare", "train", "infer", "evaluate", "visualize")


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
        # (e.g. runs/4079_{frames} -> runs/4079_300). Under `sampling: greedy` the
        # budget knob is `samples_per_class` (a per-class target), not
        # `annotation_frames` (a frame count) — substitute whichever one applies.
        data_raw = self.raw.get("data") or {}
        frames_key = "samples_per_class" if data_raw.get("sampling") == "greedy" else "annotation_frames"
        frames = str(data_raw.get(frames_key, ""))
        wd = str(self.raw["project"]["work_dir"]).replace("{frames}", frames)
        self.work = self._stamped_work_dir(self.resolve(wd))
        self.work.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.resolve(self.data.images_dir)
        self.classes = self._load_classes()
        self.steps = self._load_steps()

    def _load_steps(self) -> set[str]:
        """`project.steps: [prepare, train, ...]` toggles which steps `run_all`
        ("all" on the CLI) executes; a missing/empty key means all of them.
        Running a single step directly (`python -m pipeline.cli visualize`)
        always works regardless of this list."""
        steps = (self.raw.get("project", {}) or {}).get("steps")
        if not steps:
            return set(ALL_STEPS)
        unknown = set(steps) - set(ALL_STEPS)
        if unknown:
            raise ValueError(f"project.steps: unknown step(s) {sorted(unknown)} "
                             f"(valid: {list(ALL_STEPS)})")
        return set(steps)

    def resolve(self, p) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p)

    @staticmethod
    def _stamped_work_dir(base: Path) -> Path:
        """Prepend a `YYYYMMDD_HHMMSS_` timestamp to `base`'s folder name so each
        run keeps its own history under `runs/` instead of overwriting the last
        one. Minted once per run: `prepare`/`train`/`infer`/`evaluate`/`visualize`
        are separate CLI calls that must keep sharing one folder, so if a
        timestamped sibling for this same base name already exists, the most
        recent one is reused; a fresh timestamp is only minted when none exists.
        """
        parent, name = base.parent, base.name
        pattern = re.compile(rf"^\d{{8}}_\d{{6}}_{re.escape(name)}$")
        existing = sorted(p for p in parent.glob(f"*_{name}") if pattern.match(p.name))
        if existing:
            return existing[-1]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return parent / f"{stamp}_{name}"

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
