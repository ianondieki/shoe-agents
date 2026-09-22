"""Reusable pieces for every agent: model setup, the LangGraph loop, and agent-as-tool wrapping."""
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Annotated, Callable, Iterable, TypedDict

try:
    # Antivirus TLS scanning (Avast, AVG, Kaspersky, ESET) and corporate proxies re-sign every
    # HTTPS connection with a root that lives in the Windows certificate store. httpx verifies
    # against certifi, which does not have it, so every provider call dies with
    # CERTIFICATE_VERIFY_FAILED. truststore verifies against the OS store instead, which already
    # trusts that root. Must run before any client builds an SSL context.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tracing import TRACER, ctx, ms_since

# The prints below contain emoji. Under a supervisor, a pipe or a service the console is cp1252
# and every one of them raises UnicodeEncodeError, killing the turn. Fix it once, here.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

Approver = Callable[[str, str, dict], bool]
Candidates = list[tuple[str, Runnable]]  # (model id, tool-bound model), in the order to try them

DEFAULT_MISTRAL = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
DEFAULT_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Unbounded by default in both SDKs (Mistral 120s, Groq 60s), which freezes a turn on a stall.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))
# Opt-in history window, for providers with a tokens-per-minute cap. 0 = keep everything (the CLI default).
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "0"))


# ---------- circuit breaker ----------
# A model that answers "not available in your subscription tier" will say it again on every call.
# The trace showed 3,536ms of one turn spent re-proving a dead primary. State is module level, not
# per-agent: the orchestrator and its sub-agents are separate build_agent closures but share one key,
# and a server that rebuilds an agent per session would otherwise reset the breaker on every call.
BREAKER_TTL_S = float(os.getenv("MODEL_BREAKER_TTL_S", "600"))
FATAL_STATUS = {401, 403, 404}  # bad key, wrong tier, retired model id
# A rate limit is temporary weather, not a dead model - but re-probing a saturated model on every
# call pays its 429 round trip every turn, which is the whole cost stacking buckets is meant to
# avoid. So it gets its own short backoff: step over it for a moment, then try it again.
RATE_LIMIT_TTL_S = float(os.getenv("MODEL_RATE_LIMIT_TTL_S", "30"))
RATE_LIMITED_STATUS = {429}
_BREAKER: dict[str, dict] = {}
_BREAKER_LOCK = threading.Lock()


def _http_status(e: Exception) -> int | None:
    """Status of a provider error, whichever SDK raised it."""
    status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
    if isinstance(status, int):
        return status
    m = re.search(r"\b([45]\d\d)\b", str(e))
    return int(m.group(1)) if m else None


def _breaker_open(model: str) -> dict | None:
    """The live trip record, or None when absent or expired (expired = this call re-probes)."""
    rec = _BREAKER.get(model)
    return rec if rec and time.monotonic() < rec["until"] else None


def _breaker_trip(model: str, status: int | None, error: str) -> None:
    ttl = RATE_LIMIT_TTL_S if status in RATE_LIMITED_STATUS else BREAKER_TTL_S
    with _BREAKER_LOCK:
        _BREAKER[model] = {
            "until": time.monotonic() + ttl,
            "status": status,
            "error": error[:160],
            "at": datetime.now().isoformat(timespec="milliseconds"),
            "trips": _BREAKER.get(model, {}).get("trips", 0) + 1,
        }


def reset_breakers() -> None:
    """For tests, and for a REPL that has just corrected .env."""
    _BREAKER.clear()


def _approx_tokens(messages) -> int:
    """Roughly four characters per token: close enough to budget a history window."""
    return sum(len(str(m.content)) for m in messages) // 4


def cli_approve(agent_name: str, tool_name: str, args: dict) -> bool:
    """Default human-in-the-loop check: ask in the terminal."""
    ans = input(f"  ⚠️  [{agent_name}] allow {tool_name}({json.dumps(args)})? [y/N] ")
    return ans.strip().lower() == "y"


def build_llms(
    tools: Iterable[BaseTool],
    order: Iterable[str] = ("mistral", "groq"),
    timeout: float | None = None,
    max_retries: int = 2,
    groq_kwargs: dict | None = None,
    mistral_model: str | None = None,
    groq_model: str | None = None,
) -> Candidates:
    """Tool-bound models in the order the agent should try them.

    The agent tries them itself (rather than .with_fallbacks) so every attempt is recorded:
    which model actually answered, and why the one before it failed. `order` exists because
    the first provider sets the floor latency for every single call — a voice path wants the
    fastest one first, while the CLI can afford to prefer the cheaper one.

    An entry is "provider" or "provider:model-id". Rate limits are published PER MODEL, so
    listing two models of the same provider gives two independent token-per-minute budgets:
    ("groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b", "mistral") is three buckets, not one.
    """
    tools = list(tools)
    timeout = LLM_TIMEOUT_S if timeout is None else timeout
    out: Candidates = []
    for entry in order:
        provider, _, override = entry.partition(":")
        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI  # 3.4s import; skipped when unused

            model = override or mistral_model or DEFAULT_MISTRAL
            out.append((model, ChatMistralAI(
                model=model, temperature=0, max_retries=max_retries, timeout=int(timeout),
            ).bind_tools(tools)))
        elif provider == "groq":
            from langchain_groq import ChatGroq

            model = override or groq_model or DEFAULT_GROQ
            out.append((model, ChatGroq(
                model=model, temperature=0, max_retries=max_retries, request_timeout=timeout,
                **(groq_kwargs or {}),
            ).bind_tools(tools)))
        else:
            raise ValueError(f"unknown provider {provider!r} in {entry!r}")
    return out


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent(
    name: str,
    system,
    tools: list[BaseTool],
    sensitive: set[str] = frozenset(),
    approve: Approver = cli_approve,
    llm=None,
    recursion_limit: int | None = None,
    terminal_tools: frozenset[str] = frozenset(),
):
    """Build a tool-calling agent graph. Any agent (shoe, voice, orchestrator) is made with this.

    `system` is a string, or a callable taking the RunnableConfig so per-call context can be
    rendered without rebuilding the agent. `llm` is a single Runnable or a Candidates list.
    `terminal_tools` end the turn once they run, instead of handing control back to the model:
    for an action like hanging up, the model has already said its piece, and asking it to speak
    again after the tool only makes it repeat itself.
    Every model attempt and tool call goes through tracing.TRACER.
    """
    if llm is None:
        llms = build_llms(tools)
    elif isinstance(llm, list):
        llms = llm
    else:
        llms = [("custom", llm)]
    # A real model id is shared across agents on purpose: a model this key cannot use is dead for
    # all of them. An anonymous "custom" model is not - one agent's broken stand-in must not bench
    # every other agent's, so it is keyed by the instance.
    keys = [m if m != "custom" else f"custom:{id(c)}" for m, c in llms]
    by_name = {t.name: t for t in tools}

    def agent_node(state: State, config: RunnableConfig):
        history = state["messages"]
        if MAX_HISTORY_TOKENS:
            # The turn in progress is never trimmed: the model must see the caller's latest words and
            # the tools it already ran for them. Only earlier turns compete for what is left, and
            # they are cut on a turn boundary so no tool result is orphaned from its tool call.
            cut = max((j for j, m in enumerate(history) if m.type == "human"), default=0)
            earlier, current = history[:cut], history[cut:]
            budget = MAX_HISTORY_TOKENS - _approx_tokens(current)
            earlier = trim_messages(
                earlier, max_tokens=budget, strategy="last", start_on="human", include_system=False,
                allow_partial=False, token_counter=_approx_tokens,
            ) if earlier and budget > 0 else []
            history = earlier + current
        prompt = system(config) if callable(system) else system
        messages = [SystemMessage(prompt)] + history
        where = ctx(config)
        limit = (config or {}).get("recursion_limit", 25)
        last_error = None

        benched = [_breaker_open(k) for k in keys]
        # Never fail a turn without a real attempt. If every candidate is benched (three buckets
        # rate-limited at once), probe the one whose bench ends first - it has likely recovered,
        # and an instant error on a phone line is worse than one slow call.
        probe = (min(range(len(llms)), key=lambda j: benched[j]["until"])
                 if llms and all(benched) else None)

        for i, (model, candidate) in enumerate(llms):
            if (rec := benched[i]) is not None and i != probe:
                # Recorded, not silent: a skipped candidate stays one line in traces.jsonl.
                TRACER.event("llm", agent=name, **where, model=model, fallback=i > 0, ms=0,
                             ok=False, skipped=True, status=rec["status"],
                             error=f"skipped, breaker open: {rec['error']}", limit=limit)
                last_error = last_error or RuntimeError(f"{model}: {rec['error']}")
                continue

            t0 = time.perf_counter()
            try:
                msg = candidate.invoke(messages)
            except Exception as e:
                status, detail = _http_status(e), f"{type(e).__name__}: {e}"
                if status in FATAL_STATUS or status in RATE_LIMITED_STATUS:
                    _breaker_trip(keys[i], status, detail)
                TRACER.event("llm", agent=name, **where, model=model, fallback=i > 0,
                             ms=ms_since(t0), ok=False, skipped=False, status=status,
                             fatal=status in FATAL_STATUS, error=detail, limit=limit)
                last_error = e
                continue

            usage = getattr(msg, "usage_metadata", None) or {}
            TRACER.event("llm", agent=name, **where, model=model, fallback=i > 0,
                         ms=ms_since(t0), ok=True, skipped=False,
                         tokens_in=usage.get("input_tokens", 0), tokens_out=usage.get("output_tokens", 0),
                         tool_calls=[c["name"] for c in msg.tool_calls], limit=limit)
            return {"messages": [msg]}

        raise last_error or RuntimeError(f"[{name}] no model candidates available")

    def tools_node(state: State, config: RunnableConfig):
        where = ctx(config)
        pad = "  " * (where["depth"] + 1)
        results = []
        for call in state["messages"][-1].tool_calls:
            tname, args = call["name"], call["args"]
            print(f"{pad}🔧 [{name}] {tname}({json.dumps(args)})")
            t0 = time.perf_counter()
            ok, approved, error = True, None, None
            if tname not in by_name:
                ok, error = False, f"unknown tool {tname}"
                output = f"Unknown tool: {tname}"
            else:
                if tname in sensitive:
                    approved = approve(name, tname, args)
                if approved is False or isinstance(approved, str):
                    # A string denial carries a reason the model can act on ("confirm the price
                    # first") instead of a dead end it can only apologise for.
                    output = approved if isinstance(approved, str) else "User declined this action."
                    approved = False
                else:
                    try:
                        # config is passed down so sub-agent tools know the parent turn and thread
                        output = by_name[tname].invoke(args, config)
                    except Exception as e:
                        ok, error = False, f"{type(e).__name__}: {e}"
                        output = f"Tool error: {e}"
            print(f"{pad}   ↳ [{name}] {str(output)[:200]}")
            TRACER.event("tool", agent=name, **where, tool=tname, args=args, ms=ms_since(t0),
                         ok=ok, approved=approved, error=error, output=str(output)[:200])
            results.append(ToolMessage(str(output), tool_call_id=call["id"], name=tname))
        return {"messages": results}

    def route(state: State):
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: State):
        # End only when EVERY call in the round was terminal. Mixed with, say, place_order, the
        # model still has to hear that result and tell the customer.
        last_ai = next(m for m in reversed(state["messages"]) if m.type == "ai")
        called = {c["name"] for c in last_ai.tool_calls}
        return END if called and called <= terminal_tools else "agent"

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    if terminal_tools:
        g.add_conditional_edges("tools", after_tools, ["agent", END])
    else:
        g.add_edge("tools", "agent")
    app = g.compile(checkpointer=MemorySaver(), name=name)
    app.default_recursion_limit = recursion_limit  # read by chat_loop / callers, not by langgraph
    return app


def agent_as_tool(app, name: str, description: str) -> BaseTool:
    """Wrap a compiled agent so an orchestrator can call it like any other tool.

    The sub-agent keeps its own memory per orchestrator conversation
    (thread id = '<parent thread>:<agent name>'), so follow-ups like
    'cancel that order' still work on the next call.
    """

    @tool(name, description=description)
    def _call(request: str, config: RunnableConfig) -> str:
        parent = ctx(config)
        sub_config = {
            "configurable": {
                "thread_id": f"{parent['thread'] or 'default'}:{name}",
                "turn_id": parent["turn_id"],  # same user turn, so spend rolls up across agents
                "depth": parent["depth"] + 1,
            },
            "recursion_limit": 25,
        }
        # No try/except: a failure propagates to the caller's tools_node, the one place that
        # records tool errors before handing them back to the model.
        result = app.invoke({"messages": [HumanMessage(request)]}, sub_config)
        return result["messages"][-1].content

    return _call


def chat_loop(app, banner: str):
    """Simple terminal chat for any agent. Prints a cost/health line after every turn."""
    import uuid

    thread_id = str(uuid.uuid4())
    limit = getattr(app, "default_recursion_limit", None) or 40
    print(banner + " Type 'quit' to exit.\n")
    try:
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            turn_id = TRACER.begin_turn(text)
            config = {
                "configurable": {"thread_id": thread_id, "turn_id": turn_id, "depth": 0},
                "recursion_limit": limit,
            }
            error = None
            try:
                result = app.invoke({"messages": [HumanMessage(text)]}, config)
                print(f"Agent: {result['messages'][-1].content}\n")
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"Error: {e}\n")
            TRACER.end_turn(error)
    finally:
        TRACER.session_summary()
