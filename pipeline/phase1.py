"""Phase 1 — localization + naming. A single fine-tuned multi-class YOLO detector
finds objects and names them in one shot (no separate naming stage)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

_YOLO_SHORTHAND = {"yolo11l": "weights/yolo11l.pt", "yolo11x": "weights/yolo11x.pt"}


def yolo_base_weights(cfg) -> Path:
    """Pretrained YOLO weights to fine-tune from."""
    k = str(getattr(cfg.phase1, "model", None) or "yolo11l").lower()
    if k in _YOLO_SHORTHAND:
        return cfg.resolve(_YOLO_SHORTHAND[k])
    return cfg.resolve(getattr(cfg.phase1, "model", None) or _YOLO_SHORTHAND["yolo11l"])


def train_detector(cfg, data_yaml: Path) -> Path:
    p1 = cfg.phase1
    model = YOLO(str(yolo_base_weights(cfg)))
    model.train(
        data=str(data_yaml),
        imgsz=p1.imgsz, epochs=p1.epochs, batch=p1.batch, device=str(p1.device),
        project=str(cfg.detector_dir), name="train", exist_ok=True,
        single_cls=False, patience=p1.patience,
        close_mosaic=p1.close_mosaic, amp=p1.amp, seed=cfg.data.seed, verbose=True,
    )
    return cfg.best_weights


def _gt_boxes_px(txt: Path, W: int, H: int) -> np.ndarray:
    rows = []
    for line in Path(txt).read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            _, xc, yc, w, h = p
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            rows.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
    return np.array(rows, np.float32) if rows else np.zeros((0, 4), np.float32)


class YoloProposer:
    """Loads a trained multi-class YOLO checkpoint for inference."""

    def __init__(self, cfg, weights: Path):
        self.cfg = cfg
        self.model = YOLO(str(weights))

    def predict(self, paths):
        p = self.cfg
        return self.model.predict(paths, conf=p.infer.conf, iou=p.infer.iou_nms,
                                  imgsz=p.phase1.imgsz, max_det=p.infer.max_det,
                                  device=str(p.phase1.device), verbose=False)
