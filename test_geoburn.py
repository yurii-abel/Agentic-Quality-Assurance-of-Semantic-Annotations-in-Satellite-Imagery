"""Unit and integration tests. Scenes built by geoburn.synthetic are SYNTHETIC and only test logic."""
import json
from datetime import date
from pathlib import Path
import numpy as np
import pytest
from rasterio.transform import from_origin
from geoburn.config import load_config
from geoburn.rs import indices as ix, grid, quality, burn, sar
from geoburn.data.base import SceneRef, DataUnavailable
from geoburn.data.stac import select_pair, StacProvider
from geoburn.qa import validation as val, cmf
from geoburn.agent.controller import Controller, parse_request, infer_task_rules
from geoburn.agent.tools import REGISTRY, validate_call
from geoburn.agent.knowledge import KnowledgeBase
from geoburn.agent.evaluate import evaluate_plans, evaluate_trace
from geoburn import synthetic as syn

ROOT = Path(__file__).resolve().parents[1]


def test_indices_known_values():
    nir, swir = np.array([[0.4, 0.1]], "float32"), np.array([[0.1, 0.2]], "float32")
    n = ix.nbr(nir, swir)
    assert n[0, 0] == pytest.approx(0.6) and n[0, 1] == pytest.approx(-1 / 3, rel=1e-5)
    d = ix.dnbr(np.float32(0.6), np.float32(-0.2))
    assert d == pytest.approx(0.8)
    assert ix.rbr(d, np.float32(0.6)) == pytest.approx(0.8 / 1.601, rel=1e-5)
    assert ix.rdnbr(d, np.float32(0.64)) == pytest.approx(1.0, rel=1e-5)
    assert np.isnan(ix.nd(np.zeros((1, 1), "float32"), np.zeros((1, 1), "float32"))).all()


def test_pixel_area_projected_and_geographic():
    assert grid.pixel_area_m2(from_origin(500000, 4000000, 20, 20), "EPSG:32630", (3, 3))[0, 0] == 400
    t = from_origin(-15.76, 28.09, 0.0001, 0.0001)
    a_low = grid.pixel_area_m2(t, "EPSG:4326", (2, 2))[0, 0]
    a_high = grid.pixel_area_m2(from_origin(-15.76, 60.0, 0.0001, 0.0001), "EPSG:4326", (2, 2))[0, 0]
    assert 105 < a_low < 112 and a_high < 0.6 * a_low      # same angular size, far less ground at 60N


def test_severity_and_mmu():
    th = load_config()["thresholds"]["dnbr_severity"]
    d = np.array([[0.05, 0.2, 0.3, 0.5, 0.9]], "float32")
    assert burn.severity_classes(d, th, np.ones_like(d, bool)).tolist() == [[0, 1, 2, 3, 4]]
    m = np.zeros((60, 60), bool); m[5:8, 5:8] = True; m[20:50, 20:50] = True
    out, info = burn.clean_mask(m, np.ones_like(m), 20.0, 1.0)     # 9 px = 0.36 ha < 1 ha MMU
    assert info["regions"] == 1 and info["dropped_small"] == 1 and not out[6, 6] and out[30, 30]


def test_registration_shift_detected():
    rng = np.random.default_rng(1)
    a = rng.random((128, 128))
    assert quality.registration_shift(a, np.roll(a, (3, -2), (0, 1))) == (-3, 2)
    assert quality.registration_shift(a, a) == (0, 0)


def _ref(i, day, cc, tile="28RDS", orbit=23):
    return SceneRef(f"S2_{tile}_{i}", f"2019-{day}T11:50:00Z", cc, {}, {"grid:code": tile, "sat:relative_orbit": orbit})


def test_pair_selection_prefers_clear_close_same_tile():
    refs = [_ref(1, "07-20", 2), _ref(2, "08-09", 60), _ref(3, "08-14", 5, tile="28RDR"), _ref(4, "08-12", 4),
            _ref(5, "08-29", 80), _ref(6, "09-03", 1), _ref(7, "09-28", 0)]
    pre, post = select_pair(refs, "2019-08-17", "2019-08-25")
    assert pre.id.endswith("_4") and post.id.endswith("_6")
    with pytest.raises(DataUnavailable):
        select_pair([r for r in refs if r.datetime < "2019-08-17"], "2019-08-17")


def test_boa_offset_rule():
    assert StacProvider.boa_offset({"s2:processing_baseline": "05.09"}) == -1000
    assert StacProvider.boa_offset({"s2:processing_baseline": "02.13"}) == 0
    assert StacProvider.boa_offset({"s2:processing_baseline": "05.09", "earthsearch:boa_offset_applied": True}) == 0


def test_metrics_and_calibration():
    ref = np.zeros((20, 20), bool); ref[5:15, 5:15] = True
    pred = np.zeros_like(ref); pred[5:15, 5:10] = True
    m = val.segmentation_metrics(pred, ref, np.ones_like(ref))
    assert m["precision"] == 1 and m["recall"] == 0.5 and m["iou"] == 0.5 and m["area_error_pct"] == -50
    assert val.segmentation_metrics(ref, ref, np.ones_like(ref))["boundary_f1"] == 1
    assert val.calibration(ref.astype("float32"), ref, np.ones_like(ref))["ece"] == 0


def test_sar_change_flags_vh_drop():
    rng = np.random.default_rng(0)
    vv, vh = rng.gamma(4, 0.02, (80, 80)), rng.gamma(4, 0.005, (80, 80))
    vh2 = vh.copy(); vh2[20:50, 20:50] *= 0.3
    out = sar.sar_change_evidence(vv, vv, vh, vh2)
    assert out["change_mask"][25:45, 25:45].mean() > 0.9 and out["change_mask"][60:, 60:].mean() < 0.1


def test_tool_call_validation():
    t = REGISTRY["search_scenes"]
    assert validate_call(t, {"start": "2020-01-01", "end": "2020-02-01"}, {"bbox": [0, 0, 1, 1]}) == []
    errs = validate_call(t, {"start": 5, "bogus": 1}, {})
    assert any("missing required argument 'end'" in e for e in errs) and any("unknown argument" in e for e in errs)
    assert any("precondition" in e for e in errs) and any("should be str" in e for e in errs)


def test_request_parsing_and_task_inference():
    r = parse_request("fires in -8.3,39.6,-8.0,39.9 during the last three months", today=date(2026, 11, 19))
    assert r["bbox"] == [-8.3, 39.6, -8.0, 39.9] and r["event_start"] == "2026-08-21" and r["search_end"] == "2026-11-19"
    assert "bbox" not in parse_request("fires near Lisbon 2024-09-01")           # no geocoder: AOI is not invented
    assert infer_task_rules("is it burning right now")["task_type"] == "active_fire_detection"
    assert infer_task_rules("burn severity of the fire")["task_type"] == "burn_severity"


def test_retrieval():
    kb = KnowledgeBase()
    assert kb.workflow_for("burn_severity")[1]["steps"][0]["tool"] == "search_scenes"
    hits = kb.dual_retrieve("which dnbr threshold for burn severity with sentinel-2 swir")
    assert hits and any("thresholds" in h["text"].lower() for h in hits)


def test_agent_plan_benchmark():
    res = evaluate_plans(ROOT / "tests" / "agent_cases.yaml")
    assert res["task_acc"] == 1 and res["tool_acc"] == 1 and res["arg_acc"] == 1 and res["status_acc"] == 1, res


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    pre, post, truth, tr, crs = syn.make_pair()
    truth_path = d / "truth.tif"
    import rasterio
    with rasterio.open(truth_path, "w", driver="GTiff", height=truth.shape[0], width=truth.shape[1], count=1, dtype="uint8",
                       crs=crs, transform=tr) as dst:
        dst.write(truth.astype("uint8"), 1)
    st = Controller().run("outline the burn scar and its severity", {
        "post_path": syn.write_tif(d / "post.tif", post, tr, crs), "pre_path": syn.write_tif(d / "pre.tif", pre, tr, crs),
        "reference_path": str(truth_path), "reference_name": "synthetic truth", "out_dir": str(d / "run")})
    return st, truth, d


def test_bitemporal_pipeline_on_synthetic_scene(synthetic_run):
    st, truth, d = synthetic_run
    assert st["result"]["status"] == "completed" and st["mode"] == "bitemporal"
    m = st["validation"]["methods"]
    assert m["spectral_baseline"]["iou"] > 0.9 and m["final_hybrid"]["iou"] > 0.9
    true_ha = truth.sum() * 400 / 1e4
    assert abs(sum(f["properties"]["area_ha"] for f in st["features"]) - true_ha) / true_ha < 0.08
    assert st["severity_stats"]["high"] + st["severity_stats"]["moderate-high"] > 0
    u = st["uncertainty"]["uncertainty"]
    edge = st["uncertainty"]["boundary_band"] > 0
    assert np.nanmean(u[edge]) > np.nanmean(u[~edge])                 # uncertainty concentrates on boundaries


def test_exports_are_georeferenced(synthetic_run):
    import rasterio
    st, truth, d = synthetic_run
    run = d / "run"
    for f in ("layers.tif", "classes.tif", "burned_areas.geojson", "regions.csv", "report.md", "summary.json", "trace.json", "map/index.html"):
        assert (run / f).exists(), f
    with rasterio.open(run / "classes.tif") as src:
        assert src.crs.to_epsg() == 32628 and src.res == (20, 20) and src.read(1)[80, 110] == 1
    gj = json.loads((run / "burned_areas.geojson").read_text())
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0][:2]
    assert -16.0 < lon < -15.0 and 27.5 < lat < 28.5                  # UTM 28N grid lands in the Canaries
    assert evaluate_trace(run / "trace.json")["step_success"] == 1.0


def test_cmf_prefers_true_mask_over_shifted(synthetic_run):
    st, truth, _ = synthetic_run
    post, v = st["post"], st["spectral"]["valid"]
    judge = cmf.BAIJudge().fit(post.bands, v)
    r = cmf.arbitrate(post.bands, np.roll(truth, 25, 1), truth, v, judge, chip=40)
    assert r["preferred"] == "b" and r["delta"] > 0


def test_monotemporal_fallback_reports_no_severity(tmp_path):
    pre, post, truth, tr, crs = syn.make_pair(seed=3)
    st = Controller().run("find burned vegetation", {"post_path": syn.write_tif(tmp_path / "post.tif", post, tr, crs),
                                                    "out_dir": str(tmp_path / "run")})
    assert st["mode"] == "monotemporal" and st["severity_stats"] is None
    assert "not reported" in (tmp_path / "run" / "report.md").read_text()


def test_unusable_image_is_reported_not_mapped(tmp_path):
    pre, post, truth, tr, crs = syn.make_pair()
    for b in post:
        post[b][:, 20:] = np.nan                                       # 90% no data
    st = Controller().run("find burned vegetation", {"post_path": syn.write_tif(tmp_path / "p.tif", post, tr, crs),
                                                    "out_dir": str(tmp_path / "run")})
    assert st["result"]["status"] == "failed" and "unusable" in st["result"]["message"] and "final" not in st


def test_unsupported_and_incomplete_requests_do_not_run():
    c = Controller()
    assert c.run("show active fire hotspots burning right now in 1.0,2.0,3.0,4.0")["result"]["status"] == "not_implemented"
    s = c.run("identify all areas affected by wildfires during the last three months")
    assert s["result"]["status"] == "needs_input" and s["trace"] == []


class _FakeProvider:
    def __init__(self, refs): self.refs = refs
    def search(self, *a, **k): return self.refs


def test_no_pre_image_branches_to_post_only_then_stops_cleanly(monkeypatch):
    c = Controller()
    st_holder = {}
    import geoburn.pipeline as P
    refs = [_ref(6, "09-03", 1)]
    monkeypatch.setattr(P.stac_mod, "make_provider", lambda name, cfg: _FakeProvider(refs))
    def boom(state):                                                   # stop before any network read
        st_holder["post_ref"] = state["post_ref"].id
        raise DataUnavailable("stop here (test)")
    monkeypatch.setitem(REGISTRY, "load_scenes", type(REGISTRY["load_scenes"])("load_scenes", boom, "", {}, ("post_ref",), ("post",)))
    st = c.run("burned area in -15.76,27.96,-15.65,28.09 after the fire of 2019-08-17")
    tools = [(t["tool"], t["status"]) for t in st["trace"]]
    assert tools[:3] == [("search_scenes", "ok"), ("select_image_pair", "unavailable"), ("select_post_only", "ok")]
    assert st["trace"][1]["recovery"] == "branch:post_only" and st_holder["post_ref"].endswith("_6")
    assert st["result"]["status"] == "failed"
