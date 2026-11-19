"""Local GeoTIFF adapter (also used for the CEMS EMSR382 real-data sample and SAR RTC files)."""
from __future__ import annotations
import numpy as np
import rasterio
from .base import Scene, DataUnavailable
from ..rs.grid import grid_for_bbox, resample, approx_res_m

S2_12BAND = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]


def read_on_grid(path, grid, categorical=False, band=1):
    crs, transform, shape = grid
    with rasterio.open(path) as src:
        arr = src.read(band)
        src_res = approx_res_m(src.transform, src.crs, src.shape)
        return resample(arr if categorical else arr.astype("float32"), src.transform, src.crs, transform, crs,
                        shape, categorical=categorical, src_res_m=src_res, dst_res_m=abs(transform.a))


def load_multiband(path, band_order=S2_12BAND, res_m=20.0, scale=1.0, cloud_mask_path=None,
                   cloud_bad_values=(1, 2, 3), meta=None) -> Scene:
    """Load a multiband reflectance GeoTIFF onto a UTM analysis grid at `res_m`."""
    with rasterio.open(path) as src:
        if src.count < len(band_order):
            raise DataUnavailable(f"{path}: expected {len(band_order)} bands, found {src.count}")
        bbox = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        grid = grid_for_bbox(bbox, res_m)
        crs, transform, shape = grid
        src_res = approx_res_m(src.transform, src.crs, src.shape)
        bands = {}
        for i, name in enumerate(band_order, start=1):
            a = src.read(i).astype("float32") * scale
            a[~np.isfinite(a)] = np.nan
            bands[name] = resample(a, src.transform, src.crs, transform, crs, shape,
                                   src_res_m=src_res, dst_res_m=res_m)
        src_meta = {"source_crs": str(src.crs), "source_res_m": round(src_res, 2), "source_shape": list(src.shape)}
    valid = np.all([np.isfinite(v) for v in bands.values()], axis=0)
    valid &= bands["swir22"] > 0
    cloud_frac = None
    if cloud_mask_path:
        cm = read_on_grid(cloud_mask_path, grid, categorical=True)
        cloudy = np.isin(cm, cloud_bad_values)
        cloud_frac = float(cloudy[valid].mean()) if valid.any() else 1.0
        valid &= ~cloudy
    m = {"id": str(path), "provider": "local", "analysis_res_m": res_m, "bbox4326": list(bbox),
         "cloud_fraction": cloud_frac, **src_meta, **(meta or {})}
    return Scene(bands=bands, valid=valid, transform=transform, crs=crs, res_m=res_m, meta=m)
