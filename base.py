"""Provider-independent data model. Every Scene carries its provenance."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
NATIVE_RES_S2 = {"blue": 10, "green": 10, "red": 10, "nir": 10, "rededge1": 20, "rededge2": 20,
                 "rededge3": 20, "nir08": 20, "swir16": 20, "swir22": 20, "scl": 20}


class DataUnavailable(RuntimeError):
    """Raised instead of inventing data when nothing suitable exists."""


@dataclass
class SceneRef:
    id: str
    datetime: str
    cloud_cover: float | None
    assets: dict
    properties: dict = field(default_factory=dict)
    provider: str = ""


@dataclass
class Scene:
    bands: dict            # name -> float32 reflectance on the analysis grid (NaN = no data)
    valid: np.ndarray      # bool: usable (not nodata / cloud / shadow / snow)
    transform: object
    crs: object
    res_m: float
    meta: dict = field(default_factory=dict)
    water: np.ndarray | None = None

    @property
    def shape(self):
        return self.valid.shape

    def b(self, name):
        if name not in self.bands:
            raise DataUnavailable(f"band '{name}' missing from scene {self.meta.get('id')}")
        return np.where(self.valid, self.bands[name], np.nan)


class Provider:
    name = "base"

    def search(self, bbox, start, end, max_cloud=100.0) -> list[SceneRef]:
        raise NotImplementedError

    def load(self, ref: SceneRef, bbox, grid) -> Scene:
        raise NotImplementedError


def cache_path(cache_dir, *key) -> Path:
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{h}.npz"
