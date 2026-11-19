"""Sentinel-1 change evidence. Complementary only: backscatter also changes with rain, snow,
harvest and flooding, so a SAR change is never reported as proof of burning.

Input must be radiometrically terrain-corrected (RTC) gamma0 in linear power, same orbit direction
and relative orbit for both dates, on the analysis grid. Producing RTC from GRD (orbit files,
calibration, terrain flattening) is outside this module; use an RTC product.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def boxcar(power, size=5):
    """Speckle reduction by multilooking in linear power (never average dB)."""
    return ndi.uniform_filter(np.nan_to_num(power, nan=0.0), size=size)


def log_ratio_db(pre, post, size=5):
    a, b = boxcar(pre, size), boxcar(post, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10 * np.log10(np.maximum(b, 1e-6) / np.maximum(a, 1e-6))).astype("float32")


def sar_change_evidence(vv_pre, vv_post, vh_pre, vh_post, valid=None, size=5):
    """Return robust z-scores of VH and VV log-ratio. Forest fire typically lowers VH (volume loss)."""
    out = {}
    for name, (p0, p1) in {"vv": (vv_pre, vv_post), "vh": (vh_pre, vh_post)}.items():
        lr = log_ratio_db(p0, p1, size)
        v = lr[valid] if valid is not None else lr.ravel()
        med = np.nanmedian(v)
        mad = 1.4826 * np.nanmedian(np.abs(v - med)) + 1e-6
        out[f"{name}_logratio_db"], out[f"{name}_z"] = lr, ((lr - med) / mad).astype("float32")
    out["change_mask"] = (out["vh_z"] < -2.5) | (np.abs(out["vv_z"]) > 3.5)
    out["note"] = "SAR change is supporting evidence only; it does not identify the cause of change"
    return out
