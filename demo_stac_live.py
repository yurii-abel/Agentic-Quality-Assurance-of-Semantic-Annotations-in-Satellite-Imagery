"""Live bi-temporal demonstration (needs internet access to the STAC catalogue and its COGs).

Same event as the offline sample: Gran Canaria wildfire, ignition reported on 2019-08-17.
This script could NOT be executed where the project was written (no outbound access to the
catalogue), so treat its first run as a test. Expect roughly 20 band windows to be read; with the
default AOI that is a few tens of MB of range requests against public COGs, no credentials, no fees.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoburn.agent.controller import Controller

out = Path(__file__).resolve().parents[1] / "runs" / "gran_canaria_2019_live"
state = Controller().run(
    "Compare satellite imagery before and after the wildfire of 2019-08-17 and outline the burned areas "
    "with burn severity in bbox -15.76,27.96,-15.65,28.09",
    {"event_end": "2019-08-25", "search_end": "2019-10-06", "out_dir": str(out)})
print(json.dumps({"result": state["result"], "trace": [(t["tool"], t["status"], t.get("error")) for t in state["trace"]]}, indent=1))
