"""Faster R-CNN proposer (torchvision). Single-class ('object') detector — boxes go
to Phase 2 for naming, like the SAM/YOLOE proposers but trained.

torchvision has no `model.train()`, so this module carries its own train loop,
dataset, and checkpoint I/O. Checkpoint: <work_dir>/frcnn.pt.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          fasterrcnn_resnet50_fpn_v2,
                                          FasterRCNN, FasterRCNN_ResNet50_FPN_Weights,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _device(cfg):
    d = str(cfg.phase1.device)
    return f"cuda:{d}" if (torch.cuda.is_available() and d != "cpu") else "cpu"


def _build_model(imgsz, num_classes=2, backbone="resnet50-v2", pretrained=True):
    """backbone: resnet50 | resnet50-v2 (default) | resnet101 (bigger)."""
    bk = str(backbone).lower()
    mx = max(1333, imgsz)
    if bk in ("resnet50", "resnet50-v2", "resnet50v2"):
        v2 = "v2" in bk
        ctor = fasterrcnn_resnet50_fpn_v2 if v2 else fasterrcnn_resnet50_fpn
        w = None
        if pretrained:
            w = (FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if v2
                 else FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        m = ctor(weights=w, min_size=imgsz, max_size=mx)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        return m
    if bk in ("resnet101", "resnet152"):
        from torchvision.models import get_model_weights
        wname = f"ResNet{bk[6:]}_Weights"
        bb_w = get_model_weights(bk).DEFAULT if pretrained else None
        bb = resnet_fpn_backbone(backbone_name=bk, weights=bb_w, trainable_layers=3)
        return FasterRCNN(bb, num_classes=num_classes, min_size=imgsz, max_size=mx)
    raise ValueError(f"unknown frcnn_backbone: {backbone}")


def _gt_boxes_px(txt: Path, W: int, H: int) -> np.ndarray:
    rows = []
    for line in Path(txt).read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            _, xc, yc, w, h = p
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
            x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
            if x2 > x1 and y2 > y1:
                rows.append([x1, y1, x2, y2])
    return np.array(rows, np.float32) if rows else np.zeros((0, 4), np.float32)


def _collate(batch):
    return tuple(zip(*batch))


class _FrcnnDataset(Dataset):
    def __init__(self, cfg, stems):
        self.D, self.ext = cfg.images_dir, cfg.data.image_ext
        # keep only frames that carry at least one box (FRCNN needs non-empty targets)
        self.stems = [s for s in stems if len(_gt_boxes_px(self.D / f"{s}.txt", 1, 1))]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        im = cv2.imread(str(self.D / f"{s}.{self.ext}"))
        H, W = im.shape[:2]
        img = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        bb = _gt_boxes_px(self.D / f"{s}.txt", W, H)
        target = {"boxes": torch.as_tensor(bb, dtype=torch.float32),
                  "labels": torch.ones((len(bb),), dtype=torch.int64)}
        return img, target


def train_frcnn(cfg, stems) -> Path:
    dev = _device(cfg)
    bk = getattr(cfg.phase1, "frcnn_backbone", "resnet50-v2")
    model = _build_model(cfg.phase1.imgsz, backbone=bk).to(dev)
    ds = _FrcnnDataset(cfg, stems)
    dl = DataLoader(ds, batch_size=max(2, cfg.phase1.batch // 4), shuffle=True,
                    collate_fn=_collate, num_workers=4)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=float(getattr(cfg.phase1, "frcnn_lr", 0.005)),
                          momentum=0.9, weight_decay=5e-4)
    print(f"[frcnn] train on {len(ds)} frames, {cfg.phase1.epochs} epochs, device {dev}")
    model.train()
    for ep in range(cfg.phase1.epochs):
        tot, nb = 0.0, 0
        for imgs, targets in dl:
            imgs = [i.to(dev) for i in imgs]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[frcnn] epoch {ep + 1}/{cfg.phase1.epochs}  loss {tot / max(nb,1):.3f}", flush=True)
    ckpt = cfg.work / "frcnn.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"[frcnn] saved -> {ckpt}")
    return ckpt


class FrcnnProposer:
    def __init__(self, cfg, weights: Path):
        self.cfg = cfg
        self.dev = _device(cfg)
        self.model = _build_model(cfg.phase1.imgsz,
                                  backbone=getattr(cfg.phase1, "frcnn_backbone", "resnet50-v2"),
                                  pretrained=False)
        self.model.load_state_dict(torch.load(str(weights), map_location=self.dev))
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def propose(self, paths):
        conf = float(self.cfg.infer.conf); max_det = int(self.cfg.infer.max_det)
        out = []
        for j in range(0, len(paths), 4):                # FRCNN is heavy — small chunks
            chunk = paths[j:j + 4]
            imgs, metas = [], []
            for pth in chunk:
                im = cv2.imread(pth); H, W = im.shape[:2]
                t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                imgs.append(t.to(self.dev)); metas.append((im, H, W))
            for pr, (im, H, W) in zip(self.model(imgs), metas):
                b = pr["boxes"].cpu().numpy().astype(np.float32)
                sc = pr["scores"].cpu().numpy().astype(np.float32)
                keep = sc >= conf
                b, sc = b[keep], sc[keep]
                if len(b) > max_det:
                    o = np.argsort(-sc)[:max_det]; b, sc = b[o], sc[o]
                out.append((b, sc, im, (H, W)))
        return out
