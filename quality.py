"""Radiometric / geometric quality checks run before any index is trusted."""
from __future__ import annotations
from datetime import datetime
import numpy as np


def registration_shift(a, b, valid=None):
    """Sub-scene translation between two rasters by phase correlation. Returns (dy, dx) in pixels."""
    a, b = np.nan_to_num(a.astype("float64")), np.nan_to_num(b.astype("float64"))
    if valid is not None:
        a, b = a * valid, b * valid
    a, b = a - a.mean(), b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-12)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = dy - a.shape[0] if dy > a.shape[0] // 2 else dy
    dx = dx - a.shape[1] if dx > a.shape[1] // 2 else dx
    return int(dy), int(dx)


def scene_quality(scene) -> dict:
    v = scene.valid
    return {"valid_fraction": round(float(v.mean()), 4), "cloud_fraction": scene.meta.get("cloud_fraction"),
            "usable": bool(v.mean() >= 0.4)}


def pair_quality(pre, post) -> dict:
    joint = pre.valid & post.valid
    q = {"joint_valid_fraction": round(float(joint.mean()), 4), "warnings": []}
    if joint.mean() < 0.4:
        q["warnings"].append("less than 40% of the AOI is cloud-free in both dates")
    dy, dx = registration_shift(pre.bands["nir08"], post.bands["nir08"], joint)
    q["registration_shift_px"] = [dy, dx]
    if max(abs(dy), abs(dx)) >= 1:
        q["warnings"].append(f"pre/post images appear shifted by ({dy},{dx}) px; dNBR edges are unreliable")
    try:
        d0, d1 = (datetime.fromisoformat(s.meta["datetime"][:10]) for s in (pre, post))
        q["days_apart"] = (d1 - d0).days
        if q["days_apart"] > 90:
            q["warnings"].append("pre/post dates are >90 days apart: phenology change can mimic or mask burn signal")
    except Exception:
        q["days_apart"] = None
    q["usable"] = joint.mean() >= 0.2
    return q
