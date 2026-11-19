"""Grid utilities: resampling to one analysis grid and geographically correct pixel areas."""
from __future__ import annotations
import numpy as np
from pyproj import CRS, Geod
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

GEOD = Geod(ellps="WGS84")


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def grid_for_bbox(bbox4326, res_m: float):
    """Analysis grid (UTM, north-up) covering a lon/lat bbox. Returns crs, transform, (h, w)."""
    crs = utm_crs((bbox4326[0] + bbox4326[2]) / 2, (bbox4326[1] + bbox4326[3]) / 2)
    l, b, r, t = transform_bounds("EPSG:4326", crs, *bbox4326, densify_pts=21)
    l, b = np.floor(l / res_m) * res_m, np.floor(b / res_m) * res_m
    w, h = int(np.ceil((r - l) / res_m)), int(np.ceil((t - b) / res_m))
    return crs, Affine(res_m, 0, l, 0, -res_m, b + h * res_m), (h, w)


def pixel_area_m2(transform: Affine, crs, shape) -> np.ndarray:
    """Per-pixel ground area. Projected CRS: constant. Geographic CRS: geodesic, varies with latitude."""
    h, w = shape
    crs = CRS.from_user_input(crs)
    if not crs.is_geographic:
        return np.full(shape, abs(transform.a * transform.e), dtype="float64")
    out = np.empty(h, dtype="float64")
    lon0, dlon = transform.c, transform.a
    for r in range(h):
        top, bot = transform.f + r * transform.e, transform.f + (r + 1) * transform.e
        a, _ = GEOD.polygon_area_perimeter([lon0, lon0 + dlon, lon0 + dlon, lon0], [top, top, bot, bot])
        out[r] = abs(a)
    return np.repeat(out[:, None], w, axis=1)


def resample(src, src_transform, src_crs, dst_transform, dst_crs, dst_shape, categorical=False,
             src_res_m=None, dst_res_m=None):
    """Warp one 2-D array. Continuous: average when coarsening, else bilinear. Classes: mode/nearest."""
    coarsen = bool(src_res_m and dst_res_m and dst_res_m > src_res_m * 1.01)
    if categorical:
        method = Resampling.mode if coarsen else Resampling.nearest
        dst = np.zeros(dst_shape, dtype=src.dtype)
        reproject(src, dst, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform,
                  dst_crs=dst_crs, resampling=method)
        return dst
    dst = np.full(dst_shape, np.nan, dtype="float32")
    reproject(src.astype("float32"), dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.average if coarsen else Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def approx_res_m(transform: Affine, crs, shape) -> float:
    return float(np.sqrt(np.median(pixel_area_m2(transform, crs, (shape[0], 1)))))
