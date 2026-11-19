"""Trajectory-aware evaluation of the agent (ThinkGeo/GTA-style step metrics, on our own cases).

TaskAcc  - inferred task type equals the expected one
ToolAcc  - expected tool sequence equals the planned sequence (order-sensitive, per step)
ArgAcc   - expected key arguments are present with the expected value
StatusAcc- the plan status (ok / needs_input / not_implemented) is the expected one
Executed runs add StepSuccess (share of executed steps that ended 'ok') from trace.json files.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import yaml
from .controller import Controller


def evaluate_plans(cases_path, today=date(2026, 11, 19)):
    cases, ctl = yaml.safe_load(Path(cases_path).read_text()), Controller()
    rows, agg = [], {"task": [], "tool": [], "arg": [], "status": []}
    for c in cases:
        plan = ctl.plan(c["query"], c.get("params"), today)
        got = [s["tool"] for s in plan["steps"]]
        exp = c.get("tools", [])
        n = max(len(exp), len(got), 1)
        tool_acc = sum(a == b for a, b in zip(exp, got)) / n if (exp or got) else 1.0
        args = {(s["tool"], k): v for s in plan["steps"] for k, v in s["args"].items()}
        ea = c.get("args", {})
        arg_acc = sum(str(args.get(tuple(k.split("."))) ) == str(v) for k, v in ea.items()) / len(ea) if ea else 1.0
        r = {"query": c["query"], "task_ok": plan["task_type"] == c["task"], "status_ok": plan["status"] == c.get("status", "ok"),
             "tool_acc": round(tool_acc, 3), "arg_acc": round(arg_acc, 3), "got_task": plan["task_type"], "got_status": plan["status"]}
        rows.append(r)
        for k, v in (("task", r["task_ok"]), ("tool", tool_acc), ("arg", arg_acc), ("status", r["status_ok"])):
            agg[k].append(float(v))
    return {"n_cases": len(rows), **{f"{k}_acc": round(sum(v) / len(v), 3) for k, v in agg.items()}, "cases": rows}


def evaluate_trace(trace_path):
    t = json.loads(Path(trace_path).read_text())
    steps = t["trace"]
    return {"steps": len(steps), "step_success": round(sum(s["status"] == "ok" for s in steps) / max(len(steps), 1), 3),
            "recoveries": sum("recovery" in s for s in steps), "result": t["result"]["status"],
            "tools": [s["tool"] for s in steps]}
