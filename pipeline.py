"""Tool implementations. Each tool reads/writes the shared run state and returns a short observation
(a dict of facts the controller can check). No tool invents data: failures raise typed errors."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .config import get_logger
from .data.base import DataUnavailable
from .data import stac as stac_mod, local as local_mod
from .rs import burn, quality, grid
from .seg import models as seg
from .qa import uncertainty as unc, cmf, validation as val

log = get_logger("geoburn.pipeline")


def search_scenes(state, provider=None, start=None, end=None, max_cloud=None):
    cfg = state["cfg"]
    name = provider or cfg["default_provider"]
    prov = state.setdefault("_providers", {}).get(name) or stac_mod.make_provider(name, cfg)
    state["_providers"][name], state["provider"] = prov, name
    refs = prov.search(state["bbox"], start, end, max_cloud if max_cloud is not None else cfg["analysis"]["max_cloud_cover"])
    state["refs"] = refs
    if not refs:
        raise DataUnavailable(f"no {cfg['providers'][name]['collection']} scenes for bbox {state['bbox']} in {start}..{end}")
    return {"n_scenes": len(refs), "first": refs[0].datetime[:10], "last": refs[-1].datetime[:10], "provider": name}


def select_image_pair(state, event_start, event_end=None):
    pre, post = stac_mod.select_pair(state["refs"], event_start, event_end)
    state["pre_ref"], state["post_ref"] = pre, post
    return {"pre": pre.id, "pre_date": pre.datetime[:10], "post": post.id, "post_date": post.datetime[:10],
            "pre_cloud": pre.cloud_cover, "post_cloud": post.cloud_cover}


def select_post_only(state, after):
    post = [r for r in state.get("refs", []) if r.datetime[:10] > str(after)[:10]]
    if not post:
        raise DataUnavailable("no post-event scene available either")
    state["post_ref"] = min(post, key=lambda r: r.cloud_cover or 0)
    state.pop("pre_ref", None)
    return {"post": state["post_ref"].id, "post_date": state["post_ref"].datetime[:10]}


def load_scenes(state):
    cfg, prov = state["cfg"], state["_providers"][state["provider"]]
    g = grid.grid_for_bbox(state["bbox"], cfg["analysis"]["resolution_m"])
    state["post"] = prov.load(state["post_ref"], state["bbox"], g)
    state["pre"] = prov.load(state["pre_ref"], state["bbox"], g) if state.get("pre_ref") else None
    return {"grid_crs": str(g[0]), "shape": list(g[2]), "resolution_m": cfg["analysis"]["resolution_m"],
            "has_pre": state["pre"] is not None}


def load_local_scene(state, post_path, pre_path=None, cloud_mask=None, pre_cloud_mask=None, scale=1.0, meta=None):
    res = state["cfg"]["analysis"]["resolution_m"]
    state["post"] = local_mod.load_multiband(post_path, res_m=res, scale=scale, cloud_mask_path=cloud_mask, meta=meta)
    state["pre"] = local_mod.load_multiband(pre_path, res_m=res, scale=scale, cloud_mask_path=pre_cloud_mask) if pre_path else None
    state["bbox"] = state["post"].meta["bbox4326"]
    return {"shape": list(state["post"].shape), "resolution_m": res, "has_pre": state["pre"] is not None,
            "grid_crs": str(state["post"].crs)}


def quality_check(state):
    post, pre = state["post"], state.get("pre")
    q = {"post": quality.scene_quality(post)}
    if pre is not None:
        if pre.shape != post.shape:
            raise ValueError("pre and post scenes are not on the same grid")
        q["pre"], q["pair"] = quality.scene_quality(pre), quality.pair_quality(pre, post)
    state["quality"] = q
    if not q["post"]["usable"]:
        raise DataUnavailable(f"post-event image unusable: only {q['post']['valid_fraction']:.0%} valid pixels")
    if pre is not None and not q["pair"]["usable"]:
        raise DataUnavailable("pre/post pair unusable: too little jointly cloud-free area")
    state["mode"] = "bitemporal" if pre is not None else "monotemporal"
    return {"mode": state["mode"], **{k: v for k, v in q.items()}}


def compute_burn_indices(state):
    cfg = state["cfg"]
    s = burn.analyze_bitemporal(state["pre"], state["post"], cfg) if state["mode"] == "bitemporal" \
        else burn.analyze_monotemporal(state["post"], cfg)
    s["mask_spectral"] = (np.nan_to_num(s["p_spectral"]) > 0.5) & s["valid"]
    state["spectral"] = s
    frac = float(s["mask_spectral"][s["valid"]].mean()) if s["valid"].any() else 0.0
    return {"mode": s["mode"], "threshold": s["threshold"], "candidate_fraction": round(frac, 4)}


def segment(state, model="pixel_ensemble"):
    s, c = state["spectral"], state["cfg"]["segmenter"]
    m = seg.PixelEnsembleSegmenter(c["members"], c["samples_per_member"], c["seed"]) if model == "pixel_ensemble" \
        else seg.PrithviBurnScarSegmenter()
    out = m.predict(state["post"], state.get("pre"), s["p_spectral"], s["valid"])
    out["mask_seg"] = (np.nan_to_num(out["p_seg"]) > 0.5) & s["valid"]
    state["seg"] = out
    return {"model": out["model"], "seeds": out["seeds"], "independent_of_baseline": out["independent_of_baseline"],
            "segment_fraction": round(float(out["mask_seg"][s["valid"]].mean()), 4)}


def fuse_and_delineate(state):
    s, sg, cfg = state["spectral"], state.get("seg"), state["cfg"]
    post = state["post"]
    if sg is not None:
        p, m = seg.hybrid(s["p_spectral"], sg["p_seg"], s["valid"])
        method = "hybrid (spectral x segmentation, hysteresis)"
    else:
        p, m, method = s["p_spectral"], s["mask_spectral"], "spectral baseline only"
    m, info = burn.clean_mask(m, s["valid"], post.res_m, cfg["analysis"]["min_mapping_unit_ha"])
    area = grid.pixel_area_m2(post.transform, post.crs, post.shape)
    state["final"] = {"p": p, "mask": m, "method": method, "area_m2": area,
                      "label": ndi.label(m, structure=np.ones((3, 3)))[0]}
    return {"method": method, "regions": info["regions"], "dropped_below_mmu": info["dropped_small"],
            "burned_area_ha": round(float(area[m].sum() / 1e4), 1),
            "aoi_valid_area_ha": round(float(area[s["valid"]].sum() / 1e4), 1)}


def assess_uncertainty(state):
    s, sg, f, post = state["spectral"], state.get("seg"), state["final"], state["post"]
    masks = {"spectral": s["mask_spectral"], "segmentation": sg["mask_seg"] if sg else None, "final": f["mask"]}
    u = unc.uncertainty_layers(f["p"], masks, f["mask"], s["valid"], sg["var_seg"] if sg else None)
    layers = {"confidence": u["confidence"], "p_final": f["p"]}
    if s["mode"] == "bitemporal":
        layers["dnbr"] = s["dnbr"]
    feats = burn.polygons(f["mask"], post.transform, post.crs, f["area_m2"], layers, s.get("severity"))
    if sg:
        feats = unc.rank_regions(feats, f["label"], s["mask_spectral"], sg["mask_seg"], sg["var_seg"])
        u["possible_false_positive"] = s["mask_spectral"] & ~sg["mask_seg"] & s["valid"]
        u["possible_false_negative"] = sg["mask_seg"] & ~f["mask"] & s["valid"]
    judge = cmf.BAIJudge().fit(post.bands, s["valid"])
    arb = None
    if sg:
        arb = cmf.arbitrate(post.bands, s["mask_spectral"], sg["mask_seg"], s["valid"], judge)
        arb["a"], arb["b"] = "spectral baseline", "segmentation model"
    u["cmf_final"] = {k: v for k, v in cmf.cmf_tiled(post.bands, f["mask"], s["valid"], judge).items() if k != "per_chip"}
    u["cmf_arbitration"], u["summary"] = arb, unc.scene_confidence(u["confidence"], f["mask"], s["valid"])
    state["uncertainty"], state["features"], state["_judge"] = u, feats, judge
    return {"confidence": u["summary"], "cmf_final_F": round(u["cmf_final"]["F"], 4), "cmf_arbitration": arb,
            "n_polygons": len(feats)}


def summarize_severity(state):
    s, f = state["spectral"], state["final"]
    if s["severity"] is None:
        state["severity_stats"] = None
        return {"available": False, "reason": "burn severity needs a pre-fire image (dNBR); only burned extent is reported"}
    a, sev = f["area_m2"], s["severity"]
    stats = {burn.SEVERITY_NAMES[k]: round(float(a[f["mask"] & (sev == k)].sum() / 1e4), 1) for k in range(5)}
    state["severity_stats"] = stats
    return {"available": True, "index": s["severity_index"], "area_ha_by_class": stats,
            "thresholds": state["cfg"]["thresholds"]["dnbr_severity"]}


def validate_against_reference(state, reference_path, reference_name="reference", burned_values=(1,), strata_path=None):
    post, s, f = state["post"], state["spectral"], state["final"]
    g = (post.crs, post.transform, post.shape)
    ref = np.isin(local_mod.read_on_grid(reference_path, g, categorical=True), burned_values)
    v, a = s["valid"], f["area_m2"]
    res = {"reference": reference_name, "resolution_m": post.res_m, "methods": {
        "spectral_baseline": val.segmentation_metrics(s["mask_spectral"], ref, v, a)}}
    if state.get("seg"):
        res["methods"]["segmentation_model"] = val.segmentation_metrics(state["seg"]["mask_seg"], ref, v, a)
    res["methods"]["final_" + ("hybrid" if state.get("seg") else "spectral")] = val.segmentation_metrics(f["mask"], ref, v, a)
    res["calibration_p_final"] = val.calibration(f["p"], ref, v)
    res["uncertainty_vs_error"] = val.uncertainty_vs_error(state["uncertainty"]["uncertainty"], f["mask"], ref, v)
    res["cmf_reference_vs_final"] = cmf.arbitrate(post.bands, ref & v, f["mask"], v, state["_judge"])
    res["cmf_corruption_controls"] = cmf.corruption_controls(post.bands, ref, v, state["_judge"])
    if strata_path:
        strata = local_mod.read_on_grid(strata_path, g, categorical=True)
        res["recall_by_reference_class"] = val.recall_by_class(f["mask"], strata, v, a)
    state["validation"], state["_ref_mask"] = res, ref
    return {"f1": {k: m["f1"] for k, m in res["methods"].items()}, "iou": {k: m["iou"] for k, m in res["methods"].items()}}


def vegetation_change(state):
    """Generic extension task: NDVI loss between two dates (no cause attributed)."""
    from .rs import indices as ix
    pre, post = state["pre"], state["post"]
    v = pre.valid & post.valid
    d = np.where(v, ix.ndvi(pre.b("nir08"), pre.b("red")) - ix.ndvi(post.b("nir08"), post.b("red")), np.nan)
    t = burn.otsu(d, 0.1, 0.4) or 0.2
    p = burn._logistic(d, t, 0.04)
    state["spectral"] = {"mode": "bitemporal", "valid": v, "dndvi": d, "p_spectral": np.where(v, p, np.nan),
                         "mask_spectral": (d > t) & v, "severity": None, "severity_index": None,
                         "threshold": {"index": "dNDVI", "value": round(t, 4), "source": "scene-adaptive (Otsu)"}}
    return {"threshold": state["spectral"]["threshold"], "changed_fraction": round(float(((d > t) & v)[v].mean()), 4)}


def export_results(state, out_dir):
    from .report import export
    return export.write_all(state, out_dir)
