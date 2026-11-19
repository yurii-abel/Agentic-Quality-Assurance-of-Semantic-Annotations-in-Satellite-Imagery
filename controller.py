"""Central controller: perceive -> infer task -> retrieve workflow -> plan -> act -> check -> report.

Single agent on purpose (see docs/architecture.md). Planning is workflow-grounded: the controller
fills a retrieved expert workflow instead of free-form tool chaining, validates every call, checks
post-conditions, and follows declared fallback branches. Every step lands in a trace.
"""
from __future__ import annotations
import json, os, re, time
from datetime import date, datetime, timedelta
from pathlib import Path
from ..config import load_config, get_logger
from ..data.base import DataUnavailable
from ..seg.models import SegmenterUnavailable
from .knowledge import KnowledgeBase
from .tools import REGISTRY, validate_call

log = get_logger("geoburn.agent")
ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
BBOX = re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")
REL = re.compile(r"(?:last|past)\s+(\d+|one|two|three|four|five|six|twelve)?\s*(day|week|month|year)s?")
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "twelve": 12, None: 1}
RULES = [  # ordered: first match wins. Active fire is checked before burned area on purpose.
    ("active_fire_detection", r"active fire|burning (right )?now|currently burning|hot ?spots?|fire front|thermal anomal"),
    ("flood_mapping", r"flood|inundat"),
    ("building_change", r"building|infrastructure|construction"),
    ("burn_severity", r"severity|damage level|levels of (vegetation )?damage"),
    ("burned_area_mapping", r"burn|fire|wildfire|scar|scorch"),
    ("land_cover_classification", r"land ?cover|land ?use|classif"),
    ("vegetation_change", r"vegetation|deforest|ndvi|chang|differen|compare|before and after"),
]


def infer_task_rules(query: str) -> dict:
    q = query.lower()
    for task, pat in RULES:
        if re.search(pat, q):
            return {"task_type": task, "source": "rules"}
    return {"task_type": None, "source": "rules"}


def infer_task_llm(query: str, tasks: list[str]) -> dict | None:
    """Optional LLM task inference (Anthropic Messages API). Off unless both env vars are set.
    Not exercised in the test suite; the rule-based path is the tested default."""
    key, model = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("GEOBURN_LLM_MODEL")
    if not (key and model):
        return None
    import requests
    prompt = ("You classify remote-sensing requests. Consider sensor/modality, temporal setting and analytical "
              f"operation. Choose exactly one task type from {tasks}. Burned-area mapping, burn severity and active-fire "
              f"detection are different tasks. Request: {query!r}. Answer with JSON only: {{\"task_type\": \"...\"}}")
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": prompt}]})
        text = "".join(b.get("text", "") for b in r.json()["content"])
        t = json.loads(text[text.index("{"): text.rindex("}") + 1])["task_type"]
        return {"task_type": t, "source": f"llm:{model}"} if t in tasks else None
    except Exception as e:
        log.warning("LLM task inference failed (%s); using rules", e)
        return None


def parse_request(query: str, today: date | None = None, cfg=None) -> dict:
    """Extract AOI and time. Nothing is guessed: a missing AOI is reported as missing."""
    cfg = cfg or load_config()
    today, out = today or date.today(), {}
    m = BBOX.search(query)
    if m:
        w, s, e, n = map(float, m.groups())
        if -180 <= w < e <= 180 and -90 <= s < n <= 90:
            out["bbox"] = [w, s, e, n]
    dates = sorted(ISO.findall(query))
    pre_d, post_d = cfg["analysis"]["pre_window_days"], cfg["analysis"]["post_window_days"]
    d = lambda s: datetime.fromisoformat(s).date()
    if len(dates) >= 2:
        out.update(event_start=dates[0], event_end=dates[-1])
    elif len(dates) == 1:
        out.update(event_start=dates[0], event_end=dates[0])
    else:
        r = REL.search(query.lower())
        if r:
            n = int(r.group(1)) if (r.group(1) or "").isdigit() else NUM[r.group(1)]
            span = timedelta(days=n * {"day": 1, "week": 7, "month": 30, "year": 365}[r.group(2)])
            # window analysis: compare the state before the window with the state at its end
            out.update(event_start=str(today - span), event_end=str(today - timedelta(days=min(post_d, span.days // 2))),
                       window=True)
    if "event_start" in out:
        out["search_start"] = str(d(out["event_start"]) - timedelta(days=pre_d))
        out["search_end"] = str(min(today, d(out["event_end"]) + timedelta(days=post_d)))
    return out


class Controller:
    def __init__(self, cfg=None, kb=None, registry=None):
        self.cfg, self.kb, self.tools = cfg or load_config(), kb or KnowledgeBase(), registry or REGISTRY

    def plan(self, query: str, params: dict | None = None, today=None) -> dict:
        tasks = list(self.kb.workflows)
        inferred = infer_task_llm(query, tasks) or infer_task_rules(query)
        ctx = {**parse_request(query, today, self.cfg), **(params or {})}
        task = ctx.pop("task_type", None) or inferred["task_type"]
        if task in ("burned_area_mapping", "burn_severity") and ctx.get("post_path"):
            task = "local_burned_area"
        plan = {"query": query, "task_type": task, "task_source": inferred["source"], "context": ctx, "steps": [],
                "status": "ok", "knowledge": [k["id"] for k in self.kb.dual_retrieve(query)]}
        if task is None:
            return {**plan, "status": "unknown_task", "message": "could not map the request to a supported analysis"}
        name, wf = self.kb.workflow_for(task)
        plan["workflow"] = name
        if wf["status"] != "implemented":
            return {**plan, "status": "not_implemented", "message": f"{name}: {wf['description']}"}
        need = [] if "post_path" in ctx else [k for k in ("bbox", "event_start") if k not in ctx]
        if need:
            return {**plan, "status": "needs_input", "missing": need,
                    "message": "please provide " + " and ".join({"bbox": "an area of interest (bbox west,south,east,north)",
                                                               "event_start": "a date or date range"}[k] for k in need)}
        plan["steps"], plan["branches"] = [self._fill(s, ctx) for s in wf["steps"]], \
            {b: [self._fill(s, ctx) for s in ss] for b, ss in wf.get("branches", {}).items()}
        plan["steps"] = [s for s in plan["steps"] if s is not None]
        return plan

    @staticmethod
    def _fill(step, ctx):
        if "when" in step and not ctx.get(step["when"]):
            return None
        args = {}
        for k, v in (step.get("args") or {}).items():
            v = ctx.get(v[1:]) if isinstance(v, str) and v.startswith("$") else v
            if v is not None:
                args[k] = v
        return {"tool": step["tool"], "args": args, "on_fail": step.get("on_fail")}

    def run(self, query: str, params: dict | None = None, today=None) -> dict:
        plan = self.plan(query, params, today)
        state = {"cfg": self.cfg, "bbox": plan["context"].get("bbox"), "plan": plan, "trace": [], "query": query}
        if plan["status"] != "ok":
            state["result"] = {"status": plan["status"], "message": plan.get("message")}
            return state
        queue = list(plan["steps"])
        while queue:
            step = queue.pop(0)
            outcome = self._execute(step, state)
            if outcome == "ok":
                continue
            state["last_error"] = state["trace"][-1].get("error", "")
            branch = step.get("on_fail")
            if branch == "skip":
                continue
            if branch and branch in plan["branches"]:
                state["trace"][-1]["recovery"] = f"branch:{branch}"
                queue = list(plan["branches"][branch]) + ([] if branch.startswith(("report", "check")) else queue)
                continue
            state["result"] = {"status": "failed", "message": state["trace"][-1]["error"]}
            break
        if "result" not in state:
            state["result"] = {"status": "insufficient_evidence", "message": state["insufficient"]} \
                if state.get("insufficient") else {"status": "completed"}
        out = (state.get("outputs") or {}).get("dir") or plan["context"].get("out_dir")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "trace.json").write_text(json.dumps(
                {"plan": plan, "trace": state["trace"], "result": state["result"]}, indent=1, default=str))
        return state

    def _execute(self, step, state, retries=1):
        rec = {"tool": step["tool"], "args": step["args"], "status": "ok"}
        state["trace"].append(rec)
        tool = self.tools.get(step["tool"])
        errs = [f"unknown tool '{step['tool']}'"] if tool is None else validate_call(tool, step["args"], state)
        if errs:
            rec.update(status="invalid_call", error="; ".join(errs))
            return "error"
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                rec["observation"] = tool.fn(state, **step["args"])
                rec["seconds"] = round(time.time() - t0, 2)
                missing = [k for k in tool.produces if state.get(k) is None]
                if missing:
                    rec.update(status="postcondition_failed", error=f"tool did not produce {missing}")
                    return "error"
                log.info("%-28s ok  %s", tool.name, json.dumps(rec["observation"], default=str)[:160])
                return "ok"
            except (DataUnavailable, SegmenterUnavailable) as e:     # expected, non-retryable
                rec.update(status="unavailable", error=str(e), seconds=round(time.time() - t0, 2))
                log.warning("%-28s unavailable: %s", tool.name, e)
                return "error"
            except (OSError, ConnectionError, TimeoutError) as e:    # transient: retry once
                rec.update(status="error", error=f"{type(e).__name__}: {e}", attempts=attempt + 1)
                log.warning("%-28s transient error (attempt %d): %s", tool.name, attempt + 1, e)
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}")
                log.exception("%s failed", tool.name)
                return "error"
        return "error"
