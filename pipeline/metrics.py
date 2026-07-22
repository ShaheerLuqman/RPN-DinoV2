"""Detection metrics. IoU + a class-aware Hungarian accumulator giving
precision / recall / F1 / accuracy at configurable IoU thresholds.

`accuracy` = TP / (TP + FP + FN)  (Jaccard / critical-success-index — detection
has no true negatives). Matches the acp-agentic-workflow convention.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """gt (G,4) xyxy, pr (P,4) xyxy -> (G,P) IoU."""
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)), np.float32)
    g, p = gt[:, None, :], pr[None, :, :]
    ix1 = np.maximum(g[..., 0], p[..., 0]); iy1 = np.maximum(g[..., 1], p[..., 1])
    ix2 = np.minimum(g[..., 2], p[..., 2]); iy2 = np.minimum(g[..., 3], p[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    return inter / np.clip(ag[:, None] + ap[None, :] - inter, 1e-9, None)


class BoxEval:
    """Accumulate class-aware matches across frames; summarize at thresholds."""

    def __init__(self, thresholds=(0.5, 0.75)):
        self.thr = list(thresholds)
        self.ious: list[float] = []
        self.tp = {t: 0 for t in self.thr}
        self.fp = 0
        self.fn = 0
        self.matched = 0

    def add_frame(self, gt: list[tuple[int, list[float]]],
                  pred: list[tuple[int, list[float]]]) -> None:
        """gt/pred: list of (class_id, xyxy). Hungarian per class on IoU."""
        for cls in {c for c, _ in gt} | {c for c, _ in pred}:
            gi = [i for i, (c, _) in enumerate(gt) if c == cls]
            pj = [j for j, (c, _) in enumerate(pred) if c == cls]
            if gi and pj:
                gb = np.array([gt[i][1] for i in gi], float)
                pb = np.array([pred[j][1] for j in pj], float)
                cost = 1.0 - iou_matrix(gb, pb)
                rows, cols = linear_sum_assignment(cost)
                ug = up = 0
                for r, c in zip(rows, cols):
                    iou = 1.0 - cost[r, c]
                    if iou > 0.0:
                        self.matched += 1
                        self.ious.append(float(iou))
                        ug += 1; up += 1
                        for t in self.thr:
                            if iou >= t:
                                self.tp[t] += 1
                self.fn += len(gi) - ug
                self.fp += len(pj) - up
            else:
                self.fn += len(gi)
                self.fp += len(pj)

    def summary(self) -> dict:
        n_pred = self.matched + self.fp
        n_gt = self.matched + self.fn
        out = {
            "num_gt": n_gt, "num_pred": n_pred,
            "mean_iou": round(float(np.mean(self.ious)), 4) if self.ious else 0.0,
        }
        for t in self.thr:
            tp = self.tp[t]
            p = tp / n_pred if n_pred else 0.0
            r = tp / n_gt if n_gt else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            denom = n_pred + n_gt - tp                      # TP + FP + FN
            a = tp / denom if denom else 0.0
            k = f"{t:g}"
            out[f"precision@{k}"] = round(p, 4)
            out[f"recall@{k}"] = round(r, 4)
            out[f"f1@{k}"] = round(f, 4)
            out[f"accuracy@{k}"] = round(a, 4)
        return out
