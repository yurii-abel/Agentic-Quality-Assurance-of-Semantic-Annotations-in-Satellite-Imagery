"""Uncertainty and reliability ranking (adapted from the label-noise ranking paper; see docs)."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def _entropy(p):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype("float32")


def uncertainty_layers(p_final, masks: dict, final_mask, valid, var_seg=None):
    """Heuristic per-pixel uncertainty in [0,1]. It is a ranking signal, not a calibrated error rate."""
    ms = [m for m in masks.values() if m is not None]
    stack = np.stack(ms).astype("float32")
    disagreement = (4 * stack.mean(0) * (1 - stack.mean(0))) if len(ms) > 1 else np.zeros(final_mask.shape, "float32")
    d_in, d_out = ndi.distance_transform_edt(final_mask), ndi.distance_transform_edt(~final_mask)
    boundary = np.clip(1 - (np.where(final_mask, d_in, d_out) - 1) / 3, 0, 1).astype("float32")
    if not final_mask.any():
        boundary[:] = 0
    ens = np.clip(4 * np.nan_to_num(var_seg), 0, 1) if var_seg is not None else 0
    u = 1 - (1 - _entropy(p_final)) * (1 - disagreement) * (1 - ens)
    u = np.maximum(u, 0.5 * boundary)
    u = np.where(valid, np.clip(u, 0, 1), np.nan).astype("float32")
    return {"uncertainty": u, "confidence": (1 - u).astype("float32"), "disagreement": disagreement.astype("float32"),
            "boundary_band": boundary}


def rank_regions(features, label_img, mask_a, mask_b, var_seg):
    """Regularized Variance Ranking adapted to regions: S = IoU - (0.5 - IoU) * mean(var).

    IoU is the agreement between two methods inside the dilated region footprint (the paper uses
    prediction-vs-label IoU per image). Low S = least reliable region, review first.
    """
    for f in features:
        rid = f["properties"]["region_id"]
        foot = ndi.binary_dilation(label_img == rid, iterations=3)
        a, b = mask_a & foot, mask_b & foot
        iou = (a & b).sum() / max((a | b).sum(), 1)
        var = float(np.nanmean(var_seg[foot])) if var_seg is not None else 0.0
        f["properties"].update(method_agreement_iou=round(float(iou), 3), ensemble_variance=round(var, 4),
                               reliability_score=round(float(iou - (0.5 - iou) * var), 3))
    order = sorted(features, key=lambda f: f["properties"]["reliability_score"])
    for i, f in enumerate(order, 1):
        f["properties"]["review_priority"] = i
    return features


def scene_confidence(conf, final_mask, valid):
    c = conf[final_mask & valid]
    if c.size == 0:
        return {"mean_confidence_burned": None, "share_low_confidence": None}
    return {"mean_confidence_burned": round(float(np.nanmean(c)), 3),
            "share_low_confidence": round(float((c < 0.5).mean()), 3),
            "histogram": np.histogram(c[np.isfinite(c)], bins=10, range=(0, 1))[0].tolist()}
