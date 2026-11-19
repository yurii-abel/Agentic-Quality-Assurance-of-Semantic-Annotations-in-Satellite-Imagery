"""STAC adapter (Earth Search on AWS: no credentials; CDSE: S3 credentials from the environment).

Asset names are configuration, not code, so another STAC catalogue can be added in YAML.
"""
from __future__ import annotations
import os
from datetime import datetime
import numpy as np
from .base import Provider, Scene, SceneRef, DataUnavailable, cache_path, NATIVE_RES_S2, BANDS
from ..config import get_logger

log = get_logger("geoburn.data.stac")
# Sen2Cor scene classification: 0 nodata, 1 saturated, 3 shadow, 8/9 cloud, 10 cirrus, 11 snow; 6 water
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)


class StacProvider(Provider):
    def __init__(self, name, cfg, cache_dir, client=None):
        self.name, self.cfg, self.cache_dir, self._client = name, cfg, cache_dir, client

    def client(self):
        if self._client is None:
            from pystac_client import Client
            self._client = Client.open(self.cfg["url"])
        return self._client

    def search(self, bbox, start, end, max_cloud=100.0) -> list[SceneRef]:
        try:
            items = list(self.client().search(collections=[self.cfg["collection"]], bbox=list(bbox),
                                              datetime=f"{start}/{end}", max_items=200).items())
        except Exception as e:  # network / endpoint problems are reported, never papered over
            raise DataUnavailable(f"STAC search failed on {self.cfg['url']}: {e}") from e
        refs = []
        for it in items:
            cc = it.properties.get("eo:cloud_cover")
            if cc is not None and cc > max_cloud:
                continue
            refs.append(SceneRef(id=it.id, datetime=str(it.properties.get("datetime") or it.datetime),
                                 cloud_cover=cc, assets={k: a.href for k, a in it.assets.items()},
                                 properties=dict(it.properties), provider=self.name))
        return sorted(refs, key=lambda r: r.datetime)

    def _env(self):
        import rasterio
        if self.cfg.get("auth") == "s3":
            ak, sk = os.environ.get("CDSE_S3_ACCESS_KEY"), os.environ.get("CDSE_S3_SECRET_KEY")
            if not (ak and sk):
                raise DataUnavailable("CDSE S3 credentials missing: set CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY")
            return rasterio.Env(AWS_ACCESS_KEY_ID=ak, AWS_SECRET_ACCESS_KEY=sk, AWS_VIRTUAL_HOSTING="FALSE",
                                AWS_S3_ENDPOINT=os.environ.get("CDSE_S3_ENDPOINT", "eodata.dataspace.copernicus.eu"))
        return rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    @staticmethod
    def boa_offset(props) -> float:
        """Processing baseline >= 04.00 adds +1000 DN to L2A; some catalogues already remove it."""
        if props.get("earthsearch:boa_offset_applied"):
            return 0.0
        try:
            return -1000.0 if float(str(props.get("s2:processing_baseline", "0"))) >= 4.0 else 0.0
        except ValueError:
            return 0.0

    def load(self, ref: SceneRef, bbox, grid, bands=BANDS) -> Scene:
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        crs, transform, shape = grid
        res = abs(transform.a)
        cp = cache_path(self.cache_dir, self.name, ref.id, list(bbox), res, list(bands))
        amap = self.cfg["assets"]
        if cp.exists():
            z = np.load(cp)
            data, scl = {b: z[b] for b in bands}, z["scl"]
            log.info("cache hit %s", ref.id)
        else:
            off, data = self.boa_offset(ref.properties), {}
            with self._env():
                for b in list(bands) + ["scl"]:
                    href = ref.assets.get(amap[b])
                    if href is None:
                        raise DataUnavailable(f"asset '{amap[b]}' for band {b} missing in {ref.id}")
                    cat = b == "scl"
                    rs = Resampling.nearest if cat else (Resampling.average if NATIVE_RES_S2[b] < res else Resampling.bilinear)
                    with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=transform, width=shape[1],
                                                                height=shape[0], resampling=rs) as vrt:
                        a = vrt.read(1)
                    if cat:
                        scl = a.astype("uint8")
                    else:
                        a = a.astype("float32")
                        a[a == 0] = np.nan                      # L2A nodata
                        data[b] = np.clip((a + off) * 1e-4, 0, None)
            np.savez_compressed(cp, scl=scl, **data)
        valid = np.all([np.isfinite(v) for v in data.values()], axis=0) & ~np.isin(scl, SCL_BAD)
        meta = {"id": ref.id, "provider": self.name, "datetime": ref.datetime, "cloud_cover_scene": ref.cloud_cover,
                "cloud_fraction": float(np.isin(scl, (3, 8, 9, 10)).mean()), "analysis_res_m": res,
                "native_res_m": {b: NATIVE_RES_S2[b] for b in bands}, "catalogue": self.cfg["url"],
                "assets": {b: ref.assets.get(amap[b]) for b in bands}}
        return Scene(bands=data, valid=valid, transform=transform, crs=crs, res_m=res, meta=meta, water=scl == 6)


def select_pair(refs, event_start, event_end=None):
    """Pick (pre, post): least cloudy, closest to the event, preferring the same MGRS tile / orbit."""
    t0 = datetime.fromisoformat(str(event_start)[:10])
    t1 = datetime.fromisoformat(str(event_end or event_start)[:10])
    day = lambda r: datetime.fromisoformat(r.datetime[:10])
    pre = [r for r in refs if day(r) < t0]
    post = [r for r in refs if day(r) > t1]
    if not pre or not post:
        raise DataUnavailable(f"no usable image pair: {len(pre)} pre-event and {len(post)} post-event scenes found")
    score = lambda r, ref_t: (r.cloud_cover or 0) + 0.5 * abs((day(r) - ref_t).days)
    best_post = min(post, key=lambda r: score(r, t1))
    tile = lambda r: (r.properties.get("grid:code") or r.properties.get("s2:mgrs_tile") or r.id.split("_")[1:2])
    orbit = lambda r: r.properties.get("sat:relative_orbit")
    same = [r for r in pre if tile(r) == tile(best_post)]
    same_orbit = [r for r in same if orbit(r) is not None and orbit(r) == orbit(best_post)]
    pool = same_orbit or same or pre
    return min(pool, key=lambda r: score(r, t0)), best_post


def make_provider(name, cfg):
    p = cfg["providers"][name]
    if p["kind"] != "stac":
        raise ValueError(f"unknown provider kind {p['kind']}")
    return StacProvider(name, p, cfg["cache_dir"])
