"""Configuration: YAML defaults + environment overrides. No credentials live in files."""
from __future__ import annotations
import logging, os, re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(v):
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    path = Path(path or os.environ.get("GEOBURN_CONFIG", ROOT / "config" / "default.yaml"))
    cfg = _expand(yaml.safe_load(path.read_text()))
    for dotted, val in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    return cfg


def get_logger(name: str = "geoburn") -> logging.Logger:
    log = logging.getLogger(name)
    if not logging.getLogger("geoburn").handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
        root = logging.getLogger("geoburn")
        root.addHandler(h)
        root.setLevel(os.environ.get("GEOBURN_LOG", "INFO"))
    return log
