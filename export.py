"""Exports: GeoTIFF, GeoJSON, CSV, Markdown report and a self-contained Leaflet map."""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from ..config import ROOT
from .. import __version__


def _rgba(arr, cmap, vmin, vmax, alpha_mask):
    import matplotlib
    x = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    out = (matplotlib.colormaps[cmap](x) * 255).astype("uint8")
    out[..., 3] = np.where(alpha_mask & np.isfinite(arr), 255, 0)
    return out


def _composite(scene, names, gain=3.0):
    rgb = np.stack([np.clip(np.nan_to_num(scene.bands[n]) * gain, 0, 1) for n in names], -1)
    a = np.isfinite(scene.bands[names[0]])
    return np.dstack([(rgb ** 0.8 * 255).astype("uint8"), np.where(a, 255, 0).astype("uint8")])


def _write_overlay(rgba, transform, crs, path):
    """Warp to Web Mercator so the PNG lines up exactly under Leaflet's projection."""
    from PIL import Image
    h, w = rgba.shape[:2]
    l, b, r, t = rasterio.transform.array_bounds(h, w, transform)
    dt, dw, dh = calculate_default_transform(crs, "EPSG:3857", w, h, l, b, r, t)
    dst = np.zeros((4, dh, dw), "uint8")
    for i in range(4):
        reproject(rgba[..., i], dst[i], src_transform=transform, src_crs=crs, dst_transform=dt, dst_crs="EPSG:3857",
                  resampling=Resampling.nearest)
    Image.fromarray(np.moveaxis(dst, 0, -1), "RGBA").save(path, optimize=True)
    wl, ws, we, wn = transform_bounds("EPSG:3857", "EPSG:4326", *rasterio.transform.array_bounds(dh, dw, dt))
    return [[ws, wl], [wn, we]]


def write_all(state, out_dir):
    out = Path(out_dir)
    (out / "map").mkdir(parents=True, exist_ok=True)
    post, pre, s, f, u = state["post"], state.get("pre"), state["spectral"], state["final"], state["uncertainty"]
    sg, valid, feats = state.get("seg"), s["valid"], state["features"]
    prof = dict(driver="GTiff", height=post.shape[0], width=post.shape[1], crs=post.crs, transform=post.transform,
                compress="deflate", tiled=True)
    prov = {"post": post.meta, "pre": pre.meta if pre is not None else None}
    # --- rasters -------------------------------------------------------------------------------
    cont = {"p_final": f["p"], "p_spectral": s["p_spectral"], "uncertainty": u["uncertainty"], "confidence": u["confidence"]}
    if sg:
        cont.update(p_segmentation=sg["p_seg"], ensemble_variance=sg["var_seg"])
    for k in ("dnbr", "rdnbr", "rbr", "nbr_pre", "nbr_post", "dndvi", "char_index", "bais2"):
        if k in s:
            cont[k] = s[k]
    with rasterio.open(out / "layers.tif", "w", count=len(cont), dtype="float32", nodata=np.nan, **prof) as dst:
        for i, (k, a) in enumerate(cont.items(), 1):
            dst.write(a.astype("float32"), i)
            dst.set_band_description(i, k)
        dst.update_tags(software=f"geoburn {__version__}", analysis_resolution_m=str(post.res_m), mode=s["mode"],
                        note="analysis grid resolution; SWIR bands are 20 m native", provenance=json.dumps(prov, default=str)[:4000])
    cls = np.where(valid, f["mask"].astype("uint8"), 255)
    with rasterio.open(out / "classes.tif", "w", count=2 if s.get("severity") is not None else 1, dtype="uint8", nodata=255, **prof) as dst:
        dst.write(cls, 1)
        dst.set_band_description(1, "burned (1) / not burned (0)")
        if s.get("severity") is not None:
            dst.write(np.where(f["mask"], s["severity"], np.where(valid, 0, 255)).astype("uint8"), 2)
            dst.set_band_description(2, "dNBR severity class 0-4 inside burned mask")
    # --- vectors / tables ----------------------------------------------------------------------
    gj = {"type": "FeatureCollection", "name": "burned_areas", "features": feats,
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}}
    (out / "burned_areas.geojson").write_text(json.dumps(gj))
    cols = sorted({k for ft in feats for k in ft["properties"] if k != "severity_share"})
    with open(out / "regions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ft["properties"] for ft in feats)
    # --- summary -------------------------------------------------------------------------------
    area = f["area_m2"]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "software": f"geoburn {__version__}",
        "query": state.get("query"), "task_type": state["plan"]["task_type"], "workflow": state["plan"].get("workflow"),
        "mode": s["mode"], "method": f["method"], "analysis_resolution_m": post.res_m, "crs": str(post.crs),
        "burned_area_ha": round(float(area[f["mask"]].sum() / 1e4), 1), "regions": len(feats),
        "valid_area_ha": round(float(area[valid].sum() / 1e4), 1),
        "unobserved_area_ha": round(float(area[~valid].sum() / 1e4), 1),
        "threshold": s["threshold"], "severity_area_ha": state.get("severity_stats"),
        "confidence": u["summary"], "cmf_final": u["cmf_final"], "cmf_arbitration": u["cmf_arbitration"],
        "segmentation": {k: sg[k] for k in ("model", "members", "seeds", "independent_of_baseline")} if sg else None,
        "quality": state.get("quality"), "provenance": prov, "validation": state.get("validation"),
        "knowledge_used": state["plan"].get("knowledge"),
        "result_kind": "model prediction (not a verified observation)" if not state.get("validation")
        else "model prediction compared against a reference product",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text(_report(summary))
    # --- map -----------------------------------------------------------------------------------
    T, C, layers = post.transform, post.crs, []

    def add(key, title, rgba, group, legend=None, on=False):
        b = _write_overlay(rgba, T, C, out / "map" / f"{key}.png")
        layers.append({"key": key, "title": title, "file": f"{key}.png", "bounds": b, "group": group, "legend": legend, "on": on})

    date_of = lambda sc: (sc.meta.get("datetime") or sc.meta.get("time_range") or "date unknown")
    if pre is not None:
        add("pre_rgb", f"Pre-event true colour ({str(date_of(pre))[:10]})", _composite(pre, ["red", "green", "blue"]), "imagery")
    add("post_rgb", f"Post-event true colour ({str(date_of(post))[:21]})", _composite(post, ["red", "green", "blue"]), "imagery", on=True)
    add("post_swir", "Post-event false colour (SWIR2 / NIR / red)", _composite(post, ["swir22", "nir08", "red"], 2.5), "imagery")
    if "dnbr" in s:
        add("nbr_pre", "NBR before", _rgba(s["nbr_pre"], "RdYlGn", -0.5, 0.9, valid), "index", ["RdYlGn", -0.5, 0.9])
        add("dnbr", "dNBR", _rgba(s["dnbr"], "inferno", -0.1, 0.9, valid), "index", ["inferno", -0.1, 0.9])
    if "nbr_post" in s:
        add("nbr_post", "NBR after", _rgba(s["nbr_post"], "RdYlGn", -0.5, 0.9, valid), "index", ["RdYlGn", -0.5, 0.9])
    if "char_index" in s:
        add("char", "Char index (post-fire only)", _rgba(s["char_index"], "inferno", -1, 4, valid), "index", ["inferno", -1, 4])
    add("mask", "Burned area (final)", _rgba(f["mask"].astype("float32"), "autumn", 0, 2, f["mask"]), "result")
    if s.get("severity") is not None:
        add("severity", "Burn severity (dNBR classes)", _rgba(s["severity"].astype("float32"), "YlOrRd", 0, 4, f["mask"]), "result",
            ["YlOrRd", "low", "high"])
    add("uncertainty", "Uncertainty", _rgba(u["uncertainty"], "viridis", 0, 1, valid), "quality", ["viridis", 0, 1])
    add("confidence", "Confidence inside burned area", _rgba(u["confidence"], "RdYlBu", 0, 1, f["mask"]), "quality", ["RdYlBu", 0, 1])
    if sg:
        dis = (s["mask_spectral"] ^ sg["mask_seg"]) & valid
        add("disagreement", "Method disagreement", _rgba(dis.astype("float32"), "cool", 0, 1, dis), "quality")
    add("unobserved", "Cloud / no data (not analysed)", _rgba((~valid).astype("float32"), "gray", 0, 2, ~valid), "quality")
    if state.get("_ref_mask") is not None:
        ref = state["_ref_mask"]
        edge = ref & ~np.pad(ref, 1)[2:, 1:-1] | ref & ~np.pad(ref, 1)[:-2, 1:-1] | ref & ~np.pad(ref, 1)[1:-1, 2:] | ref & ~np.pad(ref, 1)[1:-1, :-2]
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(edge)
        add("reference", f"Reference outline ({state['validation']['reference']})", _rgba(edge.astype("float32"), "winter", 0, 1, edge), "result")
    data = {"summary": summary, "layers": layers, "geojson": gj}
    html = (ROOT / "web" / "template.html").read_text().replace("__DATA__", json.dumps(data, default=str))
    (out / "map" / "index.html").write_text(html)
    state["outputs"] = {"dir": str(out), "files": sorted(p.name for p in out.iterdir())}
    return {"dir": str(out), "files": state["outputs"]["files"], "map": str(out / "map" / "index.html")}


def _report(m):
    L = [f"# Burned-area analysis report", "", f"Generated {m['generated_utc']} by {m['software']}.", "",
         f"**Request:** {m['query']}", "", "## Result", "",
         f"- Burned area detected: **{m['burned_area_ha']:,} ha** in {m['regions']} regions "
         f"(minimum mapping unit applied; analysis grid {m['analysis_resolution_m']} m, {m['crs']}).",
         f"- Area analysed: {m['valid_area_ha']:,} ha. Not observed (cloud, shadow, no data): {m['unobserved_area_ha']:,} ha. "
         "Burned area under cloud is not included.",
         f"- This is a {m['result_kind']}.",
         f"- Mode: {m['mode']}; delineation: {m['method']}.",
         f"- Burned / unburned threshold: {m['threshold']['index']} = {m['threshold']['value']} ({m['threshold']['source']})."]
    if m["severity_area_ha"]:
        L += ["", "### Burn severity (dNBR classes, configurable thresholds)", "", "| class | area (ha) |", "|---|---|"]
        L += [f"| {k} | {v:,} |" for k, v in m["severity_area_ha"].items()]
    else:
        L += ["- Burn severity: not reported. It needs a pre-fire image; a post-fire image alone only supports extent."]
    c = m["confidence"]
    L += ["", "## Confidence and uncertainty", "",
          f"- Mean confidence inside the burned mask: {c['mean_confidence_burned']}; share of burned pixels with confidence < 0.5: {c['share_low_confidence']}.",
          "- Confidence is a heuristic (score entropy, method disagreement, ensemble variance, boundary band). It is not an accuracy estimate.",
          f"- Mask fidelity (multispectral CMF adaptation, mean over {m['cmf_final'].get('n_chips')} chips): F = {m['cmf_final']['F']:.4f}."]
    if m["cmf_arbitration"]:
        a = m["cmf_arbitration"]
        L.append(f"- CMF arbitration, {a['a']} vs {a['b']}: F = {a['F_a']} vs {a['F_b']} (prefers {a[a['preferred']]}).")
    if m["segmentation"] and not m["segmentation"]["independent_of_baseline"]:
        L.append("- The segmentation model was trained on seeds from the spectral baseline, so agreement between the two is not independent confirmation.")
    v = m.get("validation")
    if v:
        L += ["", f"## Comparison with reference: {v['reference']}", "",
              "| method | precision | recall | F1 | IoU | boundary F1 | area error % |", "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {x['precision']} | {x['recall']} | {x['f1']} | {x['iou']} | {x['boundary_f1']} | {x['area_error_pct']} |"
              for k, x in v["methods"].items()]
        L += ["", f"- Calibration of the final score: ECE = {v['calibration_p_final']['ece']}.",
              f"- Uncertainty vs error: {json.dumps(v['uncertainty_vs_error'])}"]
        if v.get("recall_by_reference_class"):
            L.append(f"- Share of each reference class mapped as burned: {json.dumps(v['recall_by_reference_class'])}")
        L.append(f"- CMF corruption controls on the reference mask: {json.dumps(v['cmf_corruption_controls'])}")
    L += ["", "## Data provenance", "", "```json", json.dumps(m["provenance"], indent=1, default=str), "```", "",
          "## Quality checks", "", "```json", json.dumps(m["quality"], indent=1, default=str), "```", "",
          "## Limitations", "",
          "- SWIR bands are 20 m; nothing here is a 10 m product.",
          "- Thresholds are defaults or scene-adaptive; they are not calibrated to field data (CBI) for this ecosystem.",
          "- Spectrally similar surfaces (dark bare soil, lava, shadow, harvested fields, water edges) can be mapped as burned; low-severity and under-canopy burns can be missed.",
          "- Burned-area extent, burn severity and active fire are different products; this report covers the first, and the second only when a pre-fire image exists.", ""]
    return "\n".join(L)
