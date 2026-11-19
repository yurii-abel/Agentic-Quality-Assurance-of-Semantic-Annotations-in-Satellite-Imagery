"""Real-data demonstration: Copernicus EMS activation EMSR382 (Gran Canaria, Spain, August 2019).

Data: Sentinel-2 L2A sample tile + CEMS delineation/grading masks published in the CEMS-Wildfire
dataset repository (github.com/MatteoM95/CEMS-Wildfire-Dataset), fetched at a pinned commit.
Download size: about 47 MB. Nothing here is synthetic.
Limit: the sample is a single post-fire least-cloud mosaic (2019-08-17..2019-10-06) with no pre-fire
image, so the agent must take the mono-temporal branch and cannot report burn severity.
"""
import json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoburn.agent.controller import Controller
from geoburn.agent.evaluate import evaluate_trace

COMMIT = "d635dea054b2756b9645659e60d1c9b09a611367"
BASE = f"https://raw.githubusercontent.com/MatteoM95/CEMS-Wildfire-Dataset/{COMMIT}/assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_"
FILES = ["S2L2A.tiff", "S2L2A.json", "CM.tif", "DEL.tif", "GRA.tif"]
root = Path(__file__).resolve().parents[1]
data, out = root / "data" / "cems_sample", root / "runs" / "emsr382_aoi01_01"
data.mkdir(parents=True, exist_ok=True)
for f in FILES:
    p = data / f"EMSR382_AOI01_01_{f}"
    if not p.exists():
        print("downloading", f)
        urllib.request.urlretrieve(BASE + f, p)
req = json.loads((data / "EMSR382_AOI01_01_S2L2A.json").read_text())["payload"]["input"]
D = str(data / "EMSR382_AOI01_01_")
state = Controller().run(
    "Find burned vegetation in this region and estimate the affected area in hectares.",
    {"post_path": D + "S2L2A.tiff", "cloud_mask": D + "CM.tif", "reference_path": D + "DEL.tif", "strata_path": D + "GRA.tif",
     "reference_name": "Copernicus EMS EMSR382 delineation (via CEMS-Wildfire dataset)", "out_dir": str(out),
     "meta": {"source": f"CEMS-Wildfire-Dataset@{COMMIT[:8]} (Sentinel-2 L2A via Sentinel Hub process API)",
              "time_range": f"{req['data'][0]['dataFilter']['timeRange']['from'][:10]}..{req['data'][0]['dataFilter']['timeRange']['to'][:10]} (leastCC mosaic)",
              "activation": "EMSR382 AOI01", "note": "all bands were delivered resampled to ~10 m; re-gridded here to 20 m UTM"}})
print(json.dumps(state["result"]))
print(json.dumps(evaluate_trace(out / "trace.json")))
v = state.get("validation")
if v:
    for k, m in v["methods"].items():
        print(f"{k:22s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} IoU={m['iou']:.3f} bF1={m['boundary_f1']:.3f} areaErr={m['area_error_pct']:+.1f}%")
print("map:", out / "map" / "index.html")
