"""Observability for every agent: one flat JSON event per LLM call, tool call and user turn.

All agents come from agent_core.build_agent, so its two nodes are the only emitters:
agent_node records every model attempt, tools_node every tool call, and chat_loop marks
the turn boundaries. Events of one user message share a turn_id across the orchestrator
and every sub-agent it called, so cost and failures roll up per turn.

Events are appended to TRACE_PATH (default traces.jsonl; empty disables the file).
Summarise a file with:

    python tracing.py [traces.jsonl]
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

# Cost per model: {"model id": ($ per 1M input tokens, $ per 1M output tokens)}.
# Copy the numbers from each provider's pricing page; a model that is missing here shows no cost.
PRICES: dict[str, tuple[float, float]] = {}

TRACE_PATH = os.getenv("TRACE_PATH", "traces.jsonl")


def new_turn_id() -> str:
    return uuid.uuid4().hex[:8]


def ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def ctx(config) -> dict:
    """Trace coordinates carried in RunnableConfig: which turn, which agent thread, how deep."""
    c = (config or {}).get("configurable", {})
    return {"turn_id": c.get("turn_id"), "thread": c.get("thread_id"), "depth": c.get("depth", 0)}


def cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    if model not in PRICES:
        return None
    per_in, per_out = PRICES[model]
    return (tokens_in * per_in + tokens_out * per_out) / 1_000_000


_JSON_MESSAGE = re.compile(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _tidy(text: str) -> str:
    """Shorten one error for the terminal, keeping the part that says what to fix.

    Providers bury the real reason in a JSON body ('...403 while fetching <url>: {"message":
    "This model is not available in your subscription tier", ...}'), which plain truncation
    cuts off, so lift that message out. The untouched text stays in the trace file.
    """
    line = " ".join(text.split())
    kind = line.split(":", 1)[0]
    if (m := _JSON_MESSAGE.search(line)) and m.group(1):
        return f"{kind}: {m.group(1)}"[:160]
    return line[:160]


def _dedupe(items: Iterable[str]) -> str:
    """'a, a, b' -> 'a ×2, b'"""
    counts = Counter(_tidy(item) for item in items)
    return ", ".join(f"{k} ×{n}" if n > 1 else k for k, n in counts.items())


def _sum_cost(llm_events: list[dict]) -> float | None:
    priced = [c for e in llm_events if (c := cost(e["model"], e["tokens_in"], e["tokens_out"])) is not None]
    return sum(priced) if priced else None


class Tracer:
    def __init__(self, path: str = TRACE_PATH):
        self.path = path
        self.session: list[dict] = []  # every event since start-up
        self.turn: list[dict] = []     # events of the turn in progress
        self._turn = None              # (turn_id, text, t0) while a turn is open
        # Events arrive from more than one thread: tool calls run in LangGraph's executor, and
        # order confirmations email from their own thread. Unserialised appends interleaved and
        # tore a JSON line in half, which then broke every reader of the trace file.
        self._lock = threading.Lock()

    def event(self, kind: str, **fields) -> dict:
        e = {"ts": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **fields}
        line = json.dumps(e, default=str) + "\n"
        with self._lock:
            self.turn.append(e)
            self.session.append(e)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        return e

    def begin_turn(self, text: str) -> str:
        turn_id = new_turn_id()
        self.turn = []
        self._turn = (turn_id, text, time.perf_counter())
        return turn_id

    def end_turn(self, error: str | None = None) -> dict:
        """Roll the turn's events up into one 'turn' event and print the one-line summary."""
        if not self._turn:
            return {}
        turn_id, text, t0 = self._turn
        self._turn = None
        llm = [e for e in self.turn if e["kind"] == "llm"]
        ok_llm = [e for e in llm if e["ok"]]
        # A candidate the circuit breaker skipped is a recorded non-event: it cost nothing and
        # is not a fresh failure, so it is counted separately rather than reported as an error.
        skipped = [e for e in llm if e.get("skipped")]
        tools = [e for e in self.turn if e["kind"] == "tool"]
        agents = Counter(e["agent"] for e in ok_llm)
        limits = {e["agent"]: e["limit"] for e in ok_llm}
        r = {
            "turn_id": turn_id, "text": text[:80], "ms": ms_since(t0), "ok": error is None, "error": error,
            "llm_calls": len(ok_llm),
            "tokens_in": sum(e["tokens_in"] for e in ok_llm),
            "tokens_out": sum(e["tokens_out"] for e in ok_llm),
            "cost": _sum_cost(ok_llm),
            "models": dict(Counter(e["model"] for e in ok_llm)),
            "agents": dict(agents),
            "fallback": sum(1 for e in ok_llm if e["fallback"]),
            "llm_errors": [f"{e['model']}: {e['error']}" for e in llm if not e["ok"] and not e.get("skipped")],
            "skipped": sorted({e["model"] for e in skipped}),
            "tools": len(tools),
            "tool_errors": [f"{e['tool']}: {e['error']}" for e in tools if not e["ok"]],
            "denied": [e["tool"] for e in tools if e.get("approved") is False],
            # recursion_limit counts graph steps and each LLM call takes two (agent + tools),
            # so roughly limit/2 calls fit; flag an agent within two calls of that.
            "near_limit": [
                f"{a}: {n} LLM calls, recursion limit {limits[a]} allows ≈{limits[a] // 2}"
                for a, n in agents.items() if n >= limits[a] // 2 - 2
            ],
        }
        self.event("turn", **r)
        self._print_turn(r)
        return r

    def _print_turn(self, r: dict):
        models = ", ".join(f"{m} ×{n}" for m, n in r["models"].items()) or "no model answered"
        line = (f"{r['ms'] / 1000:.1f}s · {r['llm_calls']} LLM calls ({models}) · "
                f"{r['tokens_in']:,} in / {r['tokens_out']:,} out · {r['tools']} tools")
        if r["cost"] is not None:
            line += f" · ${r['cost']:.4f}"
        print(f"  📊 {line}")
        if len(r["agents"]) > 1:
            print("     " + ", ".join(f"{a} ×{n}" for a, n in r["agents"].items()))
        if r["fallback"]:
            print(f"     ⚠️  {r['fallback']} call(s) served by the fallback model")
        if r.get("skipped"):
            print(f"     ⏭️  skipped (breaker open): {', '.join(r['skipped'])}")
        if r["llm_errors"]:
            print(f"     ⚠️  model errors: {_dedupe(r['llm_errors'])}")
        if r["tool_errors"]:
            print(f"     ⚠️  tool errors: {_dedupe(r['tool_errors'])}")
        if r["denied"]:
            print(f"     ✋ denied: {_dedupe(r['denied'])}")
        for warning in r["near_limit"]:
            print(f"     ⚠️  {warning}")
        if r["error"]:
            print(f"     ❌ turn failed: {r['error'][:140]}")
        print()

    def session_summary(self):
        if self.session:
            print(f"\n--- this session ---\n{summarize(self.session)}\n")


def summarize(events: list[dict]) -> str:
    """Plain-text summary of a list of events: the current session, or a whole trace file."""
    if not events:
        return "no events"
    llm = [e for e in events if e["kind"] == "llm"]
    ok_llm = [e for e in llm if e["ok"]]
    tools = [e for e in events if e["kind"] == "tool"]
    turns = [e for e in events if e["kind"] == "turn"]
    out = [f"{events[0]['ts'][:16]} → {events[-1]['ts'][:16]} · {len(turns)} turns · "
           f"{len(ok_llm)} LLM calls · {len(tools)} tool calls"]

    by_model: dict[str, Counter] = {}
    for e in ok_llm:
        m = by_model.setdefault(e["model"], Counter())
        m.update(calls=1, tokens_in=e["tokens_in"], tokens_out=e["tokens_out"], fallback=int(e["fallback"]))
    out.append(f"\n{'model':32} {'calls':>6} {'in':>10} {'out':>9} {'cost':>10}")
    for model, m in sorted(by_model.items(), key=lambda kv: -kv[1]["tokens_in"]):
        c = cost(model, m["tokens_in"], m["tokens_out"])
        cost_s = f"${c:.4f}" if c is not None else "-"
        note = f"   ← fallback ×{m['fallback']}" if m["fallback"] else ""
        out.append(f"{model:32} {m['calls']:>6} {m['tokens_in']:>10,} {m['tokens_out']:>9,} {cost_s:>10}{note}")
    # Same accounting as end_turn: a breaker skip cost nothing and is not a fresh failure.
    if llm_errors := [f"{e['model']}: {e['error']}" for e in llm if not e["ok"] and not e.get("skipped")]:
        out.append(f"model errors: {_dedupe(llm_errors)}")
    if skipped := [e["model"] for e in llm if e.get("skipped")]:
        out.append(f"skipped (breaker open): {_dedupe(skipped)}")

    denied = [e for e in tools if e.get("approved") is False]
    failed = [e for e in tools if not e["ok"]]
    out.append(f"\ntools: {len(tools) - len(denied) - len(failed)} ok · {len(failed)} errors · {len(denied)} denied")
    if failed:
        out.append("  errors: " + _dedupe(f"{e['tool']}: {e['error']}" for e in failed))
    if denied:
        out.append("  denied: " + _dedupe(e["tool"] for e in denied))

    if turns:
        out.append("\nmost expensive turns:")
        for t in sorted(turns, key=lambda t: -t["tokens_in"])[:5]:
            out.append(f"  {t['turn_id']}  {t['tokens_in']:>8,} in  {t['llm_calls']:>3} calls  {t['text']!r}")
    return "\n".join(out)


TRACER = Tracer()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else TRACE_PATH
    with open(path, encoding="utf-8") as f:
        print(summarize([json.loads(line) for line in f if line.strip()]))
