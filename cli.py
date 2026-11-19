"""Command line: python -m geoburn.cli {run,plan,doctor,eval-agent,serve}"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .config import load_config
from .agent.controller import Controller


def doctor(cfg):
    """Report what is actually reachable/installed instead of assuming it."""
    import importlib, requests
    out = {"python": sys.version.split()[0], "packages": {}, "providers": {}}
    for m in ("rasterio", "pystac_client", "sklearn", "shapely", "pyproj", "torch", "terratorch"):
        try:
            out["packages"][m] = getattr(importlib.import_module(m), "__version__", "installed")
        except ImportError:
            out["packages"][m] = None
    for name, p in cfg["providers"].items():
        try:
            out["providers"][name] = {"url": p["url"], "http": requests.get(p["url"], timeout=10).status_code, "auth": p["auth"]}
        except Exception as e:
            out["providers"][name] = {"url": p["url"], "reachable": False, "error": type(e).__name__}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geoburn")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
        p.add_argument("--event-start"), p.add_argument("--event-end")
        p.add_argument("--post"), p.add_argument("--pre"), p.add_argument("--cloud-mask"), p.add_argument("--reference")
        p.add_argument("--out", default="runs/latest"), p.add_argument("--provider")
    sub.add_parser("doctor")
    e = sub.add_parser("eval-agent"); e.add_argument("--cases", default="tests/agent_cases.yaml")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--runs", default="runs")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.cmd == "doctor":
        print(json.dumps(doctor(cfg), indent=1)); return 0
    if a.cmd == "eval-agent":
        from .agent.evaluate import evaluate_plans
        print(json.dumps(evaluate_plans(a.cases), indent=1)); return 0
    if a.cmd == "serve":
        from .server import create_app
        create_app(a.runs).run(port=a.port); return 0
    if a.provider:
        cfg["default_provider"] = a.provider
    params = {k: v for k, v in {"bbox": a.bbox, "event_start": a.event_start, "event_end": a.event_end or a.event_start,
                                "post_path": a.post, "pre_path": a.pre, "cloud_mask": a.cloud_mask,
                                "reference_path": a.reference, "out_dir": a.out}.items() if v}
    ctl = Controller(cfg)
    if a.cmd == "plan":
        print(json.dumps(ctl.plan(a.query, params), indent=1, default=str)); return 0
    if "event_start" in params:   # explicit dates need the derived search window as well
        from .agent.controller import parse_request
        params = {**parse_request(f"{params['event_start']} {params['event_end']}", cfg=cfg), **params}
    st = ctl.run(a.query, params)
    print(json.dumps({"result": st["result"], "outputs": st.get("outputs"),
                      "trace": [(t["tool"], t["status"]) for t in st["trace"]]}, indent=1, default=str))
    return 0 if st["result"]["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
