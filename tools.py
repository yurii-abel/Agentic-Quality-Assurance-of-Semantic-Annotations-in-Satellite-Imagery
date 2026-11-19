"""Dynamic toolkit: tools are registered with a schema and preconditions so the controller can
validate a call before running it (ThinkGeo: wrong tool / malformed arguments are the main failures)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from .. import pipeline as P
from ..data.base import DataUnavailable


@dataclass
class Tool:
    name: str
    fn: Callable
    description: str
    params: dict = field(default_factory=dict)      # name -> (type, required)
    requires: tuple = ()                            # keys that must exist in state
    produces: tuple = ()                            # keys that must exist afterwards
    modality: str = "optical"


def check_sar_availability(state, start, end):
    """Only checks whether Sentinel-1 acquisitions exist. It does not download or process them."""
    try:
        from pystac_client import Client
        url = state["cfg"]["providers"][state["cfg"]["default_provider"]]["url"]
        items = list(Client.open(url).search(collections=["sentinel-1-grd"], bbox=state["bbox"],
                                             datetime=f"{start}/{end}", max_items=50).items())
        return {"sentinel1_scenes": len(items), "note": "GRD needs calibration + terrain correction (RTC) before "
                "geoburn.rs.sar can use it; pass RTC GeoTIFFs via the local adapter"}
    except Exception as e:
        return {"sentinel1_scenes": None, "note": f"SAR catalogue check failed: {e}"}


def report_insufficient_evidence(state, reason):
    if state.get("last_error"):
        reason = f"{reason} (cause: {state['last_error'][:200]})"
    state["insufficient"] = reason
    return {"insufficient_evidence": True, "reason": reason}


S = str
REGISTRY = {t.name: t for t in [
    Tool("search_scenes", P.search_scenes, "Find optical scenes over the AOI in a date range (STAC).",
         {"provider": (S, False), "start": (S, True), "end": (S, True), "max_cloud": (float, False)}, ("bbox",), ("refs",)),
    Tool("select_image_pair", P.select_image_pair, "Choose the best pre-event and post-event scenes.",
         {"event_start": (S, True), "event_end": (S, False)}, ("refs",), ("pre_ref", "post_ref")),
    Tool("select_post_only", P.select_post_only, "Fallback: choose a post-event scene when no pre-event scene exists.",
         {"after": (S, True)}, ("refs",), ("post_ref",)),
    Tool("load_scenes", P.load_scenes, "Read the required bands onto the 20 m analysis grid (cached).",
         {}, ("post_ref", "bbox"), ("post",)),
    Tool("load_local_scene", P.load_local_scene, "Load user-supplied multiband GeoTIFF(s) onto the analysis grid.",
         {"post_path": (S, True), "pre_path": (S, False), "cloud_mask": (S, False), "pre_cloud_mask": (S, False),
          "scale": (float, False), "meta": (dict, False)}, (), ("post",)),
    Tool("quality_check", P.quality_check, "Cloud/no-data fractions, co-registration, date spacing; sets the mode.",
         {}, ("post",), ("quality", "mode")),
    Tool("compute_burn_indices", P.compute_burn_indices, "NBR/dNBR/RdNBR/RBR (bi-temporal) or char indices (post only).",
         {}, ("post", "mode"), ("spectral",)),
    Tool("vegetation_change", P.vegetation_change, "NDVI difference between two dates.", {}, ("pre", "post"), ("spectral",)),
    Tool("segment", P.segment, "Run a segmentation back-end to get a second candidate mask and member variance.",
         {"model": (S, False)}, ("spectral",), ("seg",)),
    Tool("fuse_and_delineate", P.fuse_and_delineate, "Fuse candidates, clean, apply minimum mapping unit, measure area.",
         {}, ("spectral",), ("final",)),
    Tool("assess_uncertainty", P.assess_uncertainty, "Uncertainty map, region reliability ranking, CMF arbitration, polygons.",
         {}, ("final",), ("uncertainty", "features")),
    Tool("summarize_severity", P.summarize_severity, "Area per dNBR severity class (bi-temporal only).", {}, ("final",), ()),
    Tool("validate_against_reference", P.validate_against_reference, "Compare with an independent reference raster.",
         {"reference_path": (S, True), "reference_name": (S, False), "burned_values": (tuple, False),
          "strata_path": (S, False)}, ("final", "uncertainty"), ("validation",)),
    Tool("export_results", P.export_results, "Write GeoJSON, GeoTIFF, CSV, report and the interactive map.",
         {"out_dir": (S, True)}, ("final", "features"), ("outputs",)),
    Tool("check_sar_availability", check_sar_availability, "Check whether Sentinel-1 scenes exist for the window.",
         {"start": (S, True), "end": (S, True)}, ("bbox",), (), "sar"),
    Tool("report_insufficient_evidence", report_insufficient_evidence, "Stop and say why no result can be given.",
         {"reason": (S, True)}, (), ("insufficient",)),
]}


def validate_call(tool: Tool, args: dict, state: dict) -> list[str]:
    errs = [f"unknown argument '{k}'" for k in args if k not in tool.params]
    for name, (typ, required) in tool.params.items():
        if name not in args or args[name] is None:
            if required:
                errs.append(f"missing required argument '{name}'")
            continue
        ok = isinstance(args[name], (int, float)) if typ is float else isinstance(args[name], (list, tuple)) \
            if typ is tuple else isinstance(args[name], typ)
        if not ok:
            errs.append(f"argument '{name}' should be {typ.__name__}, got {type(args[name]).__name__}")
    errs += [f"precondition not met: state has no '{k}'" for k in tool.requires if state.get(k) is None]
    return errs
