"""Phase 1 — localization. Pluggable proposers, selected by `phase1.proposer`:

    yolo11l / yolo11x   trained single-class YOLO detector (fine-tuned; default)
    sam                 SAM 2.1 "segment everything" -> boxes        (training-free)
    fastsam             FastSAM "segment everything" -> boxes         (training-free, fast)
    yoloe               YOLOE prompt-free open-vocab detector -> boxes (training-free)
    yoloe-visual        YOLOE visual-prompt: annotated GT boxes as exemplars (training-free)

All proposers expose `.propose(paths) -> [(boxes_xyxy, confs, orig_img_bgr, (H,W))]`,
class-agnostic (boxes only; naming is Phase 2's job). The training-free proposers skip
prepare+train.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO, RTDETR, SAM, FastSAM, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

_YOLO_SHORTHAND = {"yolo11l": "weights/yolo11l.pt", "yolo11x": "weights/yolo11x.pt"}
_TRAINING_FREE = {"sam", "fastsam", "yoloe", "yoloe-visual", "yoloe_visual"}


# ---- proposer selection ----------------------------------------------------

def proposer_kind(cfg) -> str:
    return str(getattr(cfg.phase1, "proposer", "yolo11l") or "yolo11l").lower()


def is_training_free(cfg) -> bool:
    return proposer_kind(cfg) in _TRAINING_FREE


def is_frcnn(cfg) -> bool:
    return proposer_kind(cfg) == "frcnn"


def yolo_base_weights(cfg) -> Path:
    """Pretrained YOLO weights to fine-tune from (for the yolo* proposers)."""
    k = proposer_kind(cfg)
    if k in _YOLO_SHORTHAND:
        return cfg.resolve(_YOLO_SHORTHAND[k])
    return cfg.resolve(getattr(cfg.phase1, "model", None) or _YOLO_SHORTHAND["yolo11l"])


def build_proposer(cfg, weights: Path | None = None, exemplars=None):
    k = proposer_kind(cfg)
    if k in ("sam", "fastsam"):
        if k == "fastsam":
            model_cls, default = FastSAM, "weights/FastSAM-x.pt"
        else:
            model_cls, default = SAM, "weights/sam2.1_b.pt"
        sm = getattr(cfg.phase1, "sam_model", None)
        same_family = bool(sm) and (("fastsam" in str(sm).lower()) == (k == "fastsam"))
        return SamProposer(cfg, model_cls, cfg.resolve(sm if same_family else default))
    if k == "yoloe":
        w = cfg.resolve(getattr(cfg.phase1, "yoloe_pf_model", None) or "weights/yoloe-11l-seg-pf.pt")
        return YoloeProposer(cfg, "prompt_free", w)
    if k in ("yoloe-visual", "yoloe_visual"):
        w = cfg.resolve(getattr(cfg.phase1, "yoloe_model", None) or "weights/yoloe-11l-seg.pt")
        return YoloeProposer(cfg, "visual", w, exemplars=exemplars or [])
    if k == "frcnn":
        from .frcnn import FrcnnProposer          # lazy: torchvision detection
        return FrcnnProposer(cfg, weights or (cfg.work / "frcnn.pt"))
    return YoloProposer(cfg, weights or cfg.best_weights)


# ---- training (yolo proposers only) ----------------------------------------

def is_multiclass(cfg) -> bool:
    return bool(getattr(cfg.phase1, "multiclass", False))


def train_detector(cfg, data_yaml: Path) -> Path:
    p1 = cfg.phase1
    model = (RTDETR if p1.rtdetr else YOLO)(str(yolo_base_weights(cfg)))
    model.train(
        data=str(data_yaml),
        imgsz=p1.imgsz, epochs=p1.epochs, batch=p1.batch, device=str(p1.device),
        project=str(cfg.detector_dir), name="train", exist_ok=True,
        single_cls=not is_multiclass(cfg), patience=p1.patience,
        close_mosaic=p1.close_mosaic, amp=p1.amp, seed=cfg.data.seed, verbose=True,
    )
    return cfg.best_weights


# ---- shared helpers --------------------------------------------------------

def _yolo_out(r):
    b = r.boxes.xyxy.cpu().numpy().astype(np.float32) if len(r.boxes) else np.zeros((0, 4), np.float32)
    c = r.boxes.conf.cpu().numpy().astype(np.float32) if len(r.boxes) else np.zeros((0,), np.float32)
    return b, c, r.orig_img, r.orig_shape


def _gt_boxes_px(txt: Path, W: int, H: int) -> np.ndarray:
    rows = []
    for line in Path(txt).read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            _, xc, yc, w, h = p
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            rows.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
    return np.array(rows, np.float32) if rows else np.zeros((0, 4), np.float32)


def _nms_merge(b, c, iou_thr, max_det):
    if len(b) == 0:
        return b, c
    keep = nms(torch.from_numpy(b), torch.from_numpy(c), float(iou_thr)).numpy()
    b, c = b[keep], c[keep]
    if len(b) > max_det:
        order = np.argsort(-c)[:max_det]
        b, c = b[order], c[order]
    return b, c


def build_visual_exemplars(cfg, annotated_stems, n):
    """Pick the n annotated frames with the most GT boxes to use as YOLOE visual
    prompts (all boxes collapsed to one 'object' class). Returns [(img_path, bboxes)]."""
    D, ext = cfg.images_dir, cfg.data.image_ext
    scored = []
    for s in annotated_stems:
        im = cv2.imread(str(D / f"{s}.{ext}"))
        if im is None:
            continue
        H, W = im.shape[:2]
        bb = _gt_boxes_px(D / f"{s}.txt", W, H)
        if len(bb):
            scored.append((len(bb), str(D / f"{s}.{ext}"), bb))
    scored.sort(key=lambda x: -x[0])
    return [(path, bb) for _, path, bb in scored[:max(1, n)]]


# ---- proposers -------------------------------------------------------------

class YoloProposer:
    def __init__(self, cfg, weights: Path):
        self.cfg = cfg
        self.model = (RTDETR if cfg.phase1.rtdetr else YOLO)(str(weights))

    def propose(self, paths):
        p = self.cfg
        res = self.model.predict(paths, conf=p.infer.conf, iou=p.infer.iou_nms,
                                 imgsz=p.phase1.imgsz, max_det=p.infer.max_det,
                                 device=str(p.phase1.device), verbose=False)
        return [_yolo_out(r) for r in res]


def _filter_sam(b, c, W, H, min_px, max_area_ratio, max_det):
    """Drop background (near-full-frame) and tiny masks; cap to max_det by score."""
    if len(b) == 0:
        return b, c
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]; area = w * h
    keep = (w >= min_px) & (h >= min_px) & (area <= max_area_ratio * W * H)
    b, c, area = b[keep], c[keep], area[keep]
    if len(b) > max_det:
        order = (np.argsort(-c) if c.size else np.argsort(-area))[:max_det]
        b, c = b[order], c[order]
    return b, c


class SamProposer:
    """Class-agnostic 'segment everything' -> boxes. No training."""

    def __init__(self, cfg, model_cls, weights: Path):
        self.cfg = cfg
        self.model = model_cls(str(weights))

    def propose(self, paths):
        p = self.cfg
        max_ratio = float(getattr(p.phase1, "sam_max_area_ratio", 0.9))
        min_px = int(getattr(p.phase1, "sam_min_box_px", 8))
        max_det = int(p.infer.max_det)
        out = []
        for path in paths:                              # SAM has no batched inference
            r = self.model(path, imgsz=p.phase1.imgsz, device=str(p.phase1.device),
                           verbose=False)[0]
            H, W = r.orig_shape
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                c = (r.boxes.conf.cpu().numpy().astype(np.float32)
                     if r.boxes.conf is not None else np.ones(len(b), np.float32))
            else:
                b, c = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
            b, c = _filter_sam(b, c, W, H, min_px, max_ratio, max_det)
            out.append((b, c, r.orig_img, (H, W)))
        return out


class YoloeProposer:
    """Open-vocabulary YOLOE, used class-agnostically (keep boxes, drop labels).

    mode='prompt_free' : built-in-vocab detector, no prompts.
    mode='visual'      : detect objects similar to the annotated-frame exemplars.
    """

    def __init__(self, cfg, mode, weights: Path, exemplars=None):
        self.cfg = cfg
        self.mode = mode
        self.model = YOLOE(str(weights))
        self.exemplars = exemplars or []

    def propose(self, paths):
        p = self.cfg
        conf, iou = p.infer.conf, p.infer.iou_nms
        imgsz, max_det, dev = p.phase1.imgsz, int(p.infer.max_det), str(p.phase1.device)
        if self.mode == "prompt_free":
            res = self.model.predict(paths, imgsz=imgsz, conf=conf, iou=iou,
                                     max_det=max_det, device=dev, verbose=False)
            return [_yolo_out(r) for r in res]

        # visual: run once per exemplar reference, union boxes per target, then NMS
        acc = [(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), None, None)
               for _ in paths]
        for ref_path, bb in self.exemplars:
            vp = dict(bboxes=bb, cls=np.zeros(len(bb), np.int64))
            res = self.model.predict(paths, refer_image=ref_path, visual_prompts=vp,
                                     predictor=YOLOEVPSegPredictor, imgsz=imgsz,
                                     conf=conf, device=dev, verbose=False)
            for i, r in enumerate(res):
                b, c, img, shape = _yolo_out(r)
                pb, pc, _, _ = acc[i]
                acc[i] = (np.vstack([pb, b]), np.concatenate([pc, c]), img, shape)
        return [(*_nms_merge(b, c, iou, max_det), img, shape) for (b, c, img, shape) in acc]


# Back-compat alias.
Detector = YoloProposer
