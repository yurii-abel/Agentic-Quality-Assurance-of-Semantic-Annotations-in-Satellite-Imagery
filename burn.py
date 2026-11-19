"""Burned-area mapping and burn-severity classification from spectral indices.

Burned-area mapping (where did it burn), burn severity (how strongly vegetation changed) and
active-fire detection (is it burning now; needs thermal data) are different products. This module
does the first two only.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from . import indices as ix

SEVERITY_NAMES = {0: "unburned / below threshold", 1: "low", 2: "moderate-low", 3: "moderate-high", 4: "high"}


def otsu(values, lo, hi, bins=128):
    v = values[np.isfinite(values)]
    v = v[(v >= lo - (hi - lo)) & (v <= hi + (hi - lo))]
    if v.size < 200:
        return None
    hist, edges = np.histogram(v, bins=bins)
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0, m0 = np.cumsum(p), np.cumsum(p * mids)
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0) ** 2 / (w0 * w1)
    t = float(mids[np.nanargmax(between)])
    return float(np.clip(t, lo, hi))


def _logistic(x, t, s):
    return (1 / (1 + np.exp(-(x - t) / max(s, 1e-6)))).astype("float32")


def _mad(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.0


def water_mask(scene):
    g, n = scene.bands["green"], scene.bands["nir08"]
    m = (ix.nd(g, n) > 0.0) & (n < 0.12)
    return m | scene.water if scene.water is not None else m


def analyze_bitemporal(pre, post, cfg) -> dict:
    """NBR/dNBR workflow. p_spectral is a monotone evidence score in [0,1], not a calibrated probability."""
    th = cfg["thresholds"]
    valid = pre.valid & post.valid & ~water_mask(post)
    nbr_pre, nbr_post = ix.nbr(pre.b("nir08"), pre.b("swir22")), ix.nbr(post.b("nir08"), post.b("swir22"))
    d = np.where(valid, ix.dnbr(nbr_pre, nbr_post), np.nan)
    dndvi = ix.ndvi(pre.b("nir08"), pre.b("red")) - ix.ndvi(post.b("nir08"), post.b("red"))
    t, how = th["dnbr_burned"], "configured default"
    if th.get("adaptive"):
        ta = otsu(d, *th["adaptive_range"])
        if ta is not None:
            t, how = ta, f"scene-adaptive (Otsu within {th['adaptive_range']})"
    veg_ok = _logistic(dndvi, th["dndvi_min"], 0.03)
    p = np.where(valid, _logistic(d, t, 0.04) * veg_ok, np.nan).astype("float32")
    return {"mode": "bitemporal", "valid": valid, "nbr_pre": nbr_pre, "nbr_post": nbr_post, "dnbr": d,
            "rdnbr": ix.rdnbr(d, nbr_pre), "rbr": ix.rbr(d, nbr_pre), "dndvi": dndvi.astype("float32"),
            "p_spectral": p, "threshold": {"index": "dNBR", "value": round(t, 4), "source": how},
            "severity": severity_classes(d, th["dnbr_severity"], valid), "severity_index": "dNBR"}


def analyze_monotemporal(post, cfg) -> dict:
    """Post-fire-only fallback. Weaker evidence: dark soil, shadow and bare rock imitate char."""
    valid = post.valid & ~water_mask(post)
    nbr_post = ix.nbr(post.b("nir08"), post.b("swir22"))
    b2 = ix.bais2(post.b("red"), post.b("rededge2"), post.b("rededge3"), post.b("nir08"), post.b("swir22"))
    mi = ix.mirbi(post.b("swir16"), post.b("swir22"))
    z = lambda a: (a - np.nanmedian(a[valid])) / max(_mad(a[valid]), 1e-6)
    char = np.where(valid, (z(-nbr_post) + z(b2) + z(mi)) / 3, np.nan).astype("float32")
    lo, hi = np.nanpercentile(char, [50, 99])
    t = otsu(char, lo, hi) or float(np.nanpercentile(char, 85))
    p = np.where(valid, _logistic(char, t, 0.35), np.nan).astype("float32")
    return {"mode": "monotemporal", "valid": valid, "nbr_post": nbr_post, "bais2": b2, "mirbi": mi,
            "char_index": char, "p_spectral": p,
            "threshold": {"index": "char composite z(-NBR, BAIS2, MIRBI)", "value": round(t, 4),
                          "source": "scene-adaptive (Otsu); no pre-fire image"},
            "severity": None, "severity_index": None}


def severity_classes(d, th, valid):
    edges = [th["unburned"], th["low"], th["moderate_low"], th["moderate_high"]]
    cls = np.digitize(np.nan_to_num(d, nan=-9), edges).astype("uint8")
    cls[~valid] = 255
    return cls


def clean_mask(mask, valid, res_m, mmu_ha):
    """Remove speckle, close pin-holes, drop objects below the minimum mapping unit."""
    m = ndi.binary_opening(mask & valid, structure=np.ones((3, 3)))
    m = ndi.binary_closing(m, structure=np.ones((3, 3))) & valid
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m, {"regions": 0, "dropped_small": 0}
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    min_px = mmu_ha * 1e4 / (res_m * res_m)
    keep = np.concatenate([[False], sizes >= min_px])
    out = keep[lab]
    holes = ndi.binary_fill_holes(out) & ~out & valid
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, index=np.arange(1, hn + 1))
        out |= np.concatenate([[False], hs < min_px])[hl]
    return out, {"regions": int(keep.sum()), "dropped_small": int((~keep[1:]).sum())}


def hysteresis(p, low, high, valid):
    """Keep pixels above `low` only if connected to a pixel above `high`."""
    lo = (np.nan_to_num(p) > low) & valid
    lab, n = ndi.label(lo, structure=np.ones((3, 3)))
    if n == 0:
        return lo
    strong = ndi.maximum(np.nan_to_num(p), lab, index=np.arange(1, n + 1)) > high
    return np.concatenate([[False], strong])[lab]


def polygons(mask, transform, crs, area_m2, layers=None, severity=None):
    """Vectorise regions. Geometry is returned in EPSG:4326; area is summed from true pixel areas."""
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    import shapely
    from pyproj import Transformer
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_ll = lambda xy: np.column_stack(tf.transform(xy[:, 0], xy[:, 1]))
    geoms = {}
    for geom, val in shapes(lab.astype("int32"), mask=mask, transform=transform, connectivity=8):
        geoms.setdefault(int(val), []).append(shape(geom))
    feats = []
    for rid in range(1, n + 1):
        sel = lab == rid
        g = shapely.transform(unary_union(geoms[rid]), to_ll)
        props = {"region_id": rid, "area_ha": round(float(area_m2[sel].sum() / 1e4), 2), "pixels": int(sel.sum())}
        for name, arr in (layers or {}).items():
            props[f"mean_{name}"] = round(float(np.nanmean(arr[sel])), 4)
        if severity is not None:
            s = severity[sel]
            props["severity_share"] = {SEVERITY_NAMES[k]: round(float((s == k).mean()), 3) for k in range(5)}
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": props})
    return sorted(feats, key=lambda f: -f["properties"]["area_ha"])
