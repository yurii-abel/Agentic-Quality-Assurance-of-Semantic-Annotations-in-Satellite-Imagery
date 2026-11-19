"""Spectral indices. Inputs are surface reflectance in [0, 1]; NaN marks invalid pixels."""
from __future__ import annotations
import numpy as np


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.abs(a + b) < 1e-6, np.nan, out).astype("float32")


def nbr(nir, swir22):      # Normalized Burn Ratio (Key & Benson; USGS)
    return nd(nir, swir22)


def nbr2(swir16, swir22):  # sensitive to post-fire moisture / recovery
    return nd(swir16, swir22)


def ndvi(nir, red):
    return nd(nir, red)


def dnbr(nbr_pre, nbr_post):
    return (nbr_pre - nbr_post).astype("float32")


def rdnbr(dnbr_, nbr_pre):
    """Relativized dNBR (Miller & Thode 2007), unscaled form: dNBR / sqrt(|NBR_pre|)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (dnbr_ / np.sqrt(np.maximum(np.abs(nbr_pre), 1e-3))).astype("float32")


def rbr(dnbr_, nbr_pre):
    """Relativized Burn Ratio (Parks et al. 2014): dNBR / (NBR_pre + 1.001)."""
    return (dnbr_ / (nbr_pre + 1.001)).astype("float32")


def mirbi(swir16, swir22):
    """Mid-Infrared Burn Index (Trigg & Flasse 2001)."""
    return (10 * swir22 - 9.8 * swir16 + 2).astype("float32")


def bais2(red, re2, re3, nir08, swir22):
    """Burned Area Index for Sentinel-2 (Filipponi 2018). Mono-temporal char signal."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1 - np.sqrt(np.maximum(re2 * re3 * nir08, 0) / np.maximum(red, 1e-4))
        b = (swir22 - nir08) / np.sqrt(np.maximum(swir22 + nir08, 1e-4)) + 1
    return (a * b).astype("float32")
