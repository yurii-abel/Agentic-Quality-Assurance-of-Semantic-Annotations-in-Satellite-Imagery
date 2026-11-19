"""Minimal backend: POST /api/analyze runs the agent; finished runs are served as static files."""
from __future__ import annotations
import re, time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, abort
from .agent.controller import Controller


def create_app(runs_dir="runs"):
    app, runs = Flask(__name__), Path(runs_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)

    @app.post("/api/analyze")
    def analyze():
        body = request.get_json(force=True)
        if not isinstance(body.get("query"), str):
            abort(400, "field 'query' is required")
        run_id = time.strftime("%Y%m%d-%H%M%S")
        params = {k: body[k] for k in ("bbox", "event_start", "event_end") if k in body}   # no file paths from clients
        st = Controller().run(body["query"], {**params, "out_dir": str(runs / run_id)})
        return jsonify({"run": run_id, "result": st["result"], "plan_status": st["plan"]["status"],
                        "map": f"/runs/{run_id}/map/index.html" if st.get("outputs") else None,
                        "trace": [{"tool": t["tool"], "status": t["status"]} for t in st["trace"]]})

    @app.get("/runs/<run_id>/<path:rest>")
    def files(run_id, rest):
        if not re.fullmatch(r"[\w\-]+", run_id):
            abort(404)
        return send_from_directory(runs / run_id, rest)

    @app.get("/")
    def index():
        items = sorted(p.name for p in runs.iterdir() if (p / "map" / "index.html").exists())
        return "<h1>GeoBurn runs</h1>" + "".join(f'<p><a href="/runs/{i}/map/index.html">{i}</a></p>' for i in items)
    return app
