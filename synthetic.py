"""SYNTHETIC scenes for unit tests only. Never used for, or presented as, a real detection."""
from __future__ import annotations
import numpy as np
import rasterio
from rasterio.transform import from_origin

ORDER = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
VEG = dict(coastal=.03, blue=.03, green=.06, red=.04, rededge1=.10, rededge2=.25, rededge3=.32, nir=.36, nir08=.38, nir09=.38, swir16=.18, swir22=.08)
CHAR = dict(coastal=.04, blue=.04, green=.05, red=.07, rededge1=.08, rededge2=.10, rededge3=.11, nir=.12, nir08=.13, nir09=.13, swir16=.22, swir22=.20)


def make_pair(shape=(160, 200), res=20.0, seed=0, origin=(440000.0, 3110000.0)):
    """Vegetated scene; an elliptical patch burns between pre and post. Returns arrays + truth mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    truth = ((yy - 80) / 38.0) ** 2 + ((xx - 110) / 55.0) ** 2 < 1
    sev = np.clip(1 - np.sqrt(((yy - 80) / 38.0) ** 2 + ((xx - 110) / 55.0) ** 2), 0, 1) * 0.6 + 0.4
    pre, post = {}, {}
    for b in ORDER:
        base = VEG[b] * (1 + 0.08 * rng.standard_normal(shape))
        pre[b] = base.astype("float32")
        burned = VEG[b] + (CHAR[b] - VEG[b]) * sev
        post[b] = np.where(truth, burned * (1 + 0.08 * rng.standard_normal(shape)), base * (1 + 0.03 * rng.standard_normal(shape))).astype("float32")
    return pre, post, truth, from_origin(origin[0], origin[1], res, res), "EPSG:32628"


def write_tif(path, bands, transform, crs):
    arr = np.stack([bands[b] for b in ORDER]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=12, dtype="float32",
                       crs=crs, transform=transform) as dst:
        dst.write(arr)
    return str(path)
