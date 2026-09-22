"""Exercise the tracing layer end-to-end with scripted fake models: no API keys, no network."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
os.makedirs(SCRATCH, exist_ok=True)
TRACE = os.path.join(SCRATCH, "traces.jsonl")
DB = os.path.join(SCRATCH, "test.db")
for p in (TRACE, DB):
    if os.path.exists(p):
        os.remove(p)
os.environ["TRACE_PATH"] = TRACE
os.environ["DB_PATH"] = DB
sys.path.insert(0, ROOT)

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent_core import chat_loop
from orchestrator import build_orchestrator
from shoe_agent import build_shoe_agent, shoe_tool


def tc(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def ai(content="", calls=(), tokens=(100, 10)):
    return AIMessage(content=content, tool_calls=list(calls),
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                     "total_tokens": sum(tokens)})


def rate_limited(_):
    raise RuntimeError("429 Too Many Requests")


# --- turn 1: orchestrator -> shoe agent; the primary model always fails so every call falls back ---
shoe_script = [
    ai(calls=[tc("get_customer_info", {"customer_name": "Jane"}, 1)]),
    ai(calls=[tc("check_shoe_inventory", {"activity": "Running"}, 2)]),
    ai(calls=[tc("place_order", {"shoe_id": 101, "customer_id": 1}, 3)]),
    ai(calls=[tc("send_email", {"to": "jane@example.com", "subject": "Order", "body": "Placed"}, 4)]),
    ai(calls=[tc("place_order", {"shoe_id": "abc", "customer_id": 1}, 5)]),  # pydantic validation error
    ai(calls=[tc("frobnicate", {}, 6)]),                                       # hallucinated tool
    ai("Order 1 placed for Jane Doe (customer 1), shoe 101. Email declined.", tokens=(500, 40)),
]
shoe = shoe_tool(
    approve=lambda agent, tool, args: tool != "send_email",
    llm=[("mistral-large-latest", RunnableLambda(rate_limited)),
         ("openai/gpt-oss-120b", FakeMessagesListChatModel(responses=shoe_script))],
)
orch_script = [
    ai(calls=[tc("shoe_store_agent", {"request": "Jane Doe wants running shoes; order the cheapest and email her"}, 10)]),
    ai("Done: order 1 placed for Jane; the email was declined.", tokens=(300, 30)),
]
orch = build_orchestrator(sub_agents=[shoe], llm=FakeMessagesListChatModel(responses=orch_script))

sys.stdin = io.StringIO("Jane Doe needs running shoes, order the cheapest and email her\nquit\n")
chat_loop(orch, "TEST orchestrator")

# --- turn 2: shoe agent alone; every model fails, so the turn itself fails ---
dead = build_shoe_agent(llm=[("mistral-large-latest", RunnableLambda(rate_limited))])
sys.stdin = io.StringIO("hello\nquit\n")
chat_loop(dead, "TEST dead agent")

# --- assertions on the JSONL ---
with open(TRACE, encoding="utf-8") as f:
    events = [json.loads(line) for line in f]
turns = [e for e in events if e["kind"] == "turn"]
assert len(turns) == 2, turns
t1, t2 = turns
llm1 = [e for e in events if e["kind"] == "llm" and e["turn_id"] == t1["turn_id"]]
tools1 = [e for e in events if e["kind"] == "tool" and e["turn_id"] == t1["turn_id"]]

# every event of turn 1 shares one turn id, across both agents
assert {e["turn_id"] for e in llm1 + tools1} == {t1["turn_id"]}
assert t1["agents"] == {"orchestrator": 2, "shoe_store_agent": 7}, t1["agents"]
# fallback: 7 shoe calls served by groq, each preceded by a recorded mistral failure
assert t1["fallback"] == 7 and t1["llm_calls"] == 9, (t1["fallback"], t1["llm_calls"])
assert sum(1 for e in llm1 if not e["ok"]) == 7
assert all("429" in e["error"] for e in llm1 if not e["ok"])
assert {e["model"] for e in llm1 if e["ok"]} == {"custom", "openai/gpt-oss-120b"}
# tokens roll up across agents: 6x100 + 500 (shoe) + 100 + 300 (orchestrator)
assert t1["tokens_in"] == 1500, t1["tokens_in"]
assert t1["tokens_out"] == 6 * 10 + 40 + 10 + 30, t1["tokens_out"]
# depth and recursion limits
assert {e["depth"] for e in llm1 if e["agent"] == "orchestrator"} == {0}
assert {e["depth"] for e in llm1 if e["agent"] == "shoe_store_agent"} == {1}
assert {e["limit"] for e in llm1 if e["ok"] and e["agent"] == "orchestrator"} == {40}
assert {e["limit"] for e in llm1 if e["ok"] and e["agent"] == "shoe_store_agent"} == {25}
# thread ids encode the agent path
orch_thread = next(e["thread"] for e in llm1 if e["agent"] == "orchestrator")
assert {e["thread"] for e in llm1 if e["agent"] == "shoe_store_agent"} == {f"{orch_thread}:shoe_store_agent"}
# tools: denial, validation error, unknown tool, and the sub-agent call itself
assert t1["denied"] == ["send_email"], t1["denied"]
# both place_order calls were sensitive and approved (the second then failed validation)
assert [e["tool"] for e in tools1 if e.get("approved") is True] == ["place_order", "place_order"]
assert [e.get("approved") for e in tools1 if e["tool"] in ("get_customer_info", "check_shoe_inventory")] == [None, None]
errs = [e for e in tools1 if not e["ok"]]
assert [e["tool"] for e in errs] == ["place_order", "frobnicate"], errs
assert "validation" in errs[0]["error"].lower(), errs[0]["error"]
assert len(t1["tool_errors"]) == 2
sub = next(e for e in tools1 if e["tool"] == "shoe_store_agent")
assert sub["ok"] and sub["depth"] == 0 and sub["ms"] >= 0
assert t1["ok"] and t1["error"] is None
# turn 2
assert not t2["ok"] and "429" in t2["error"], t2
assert t2["llm_calls"] == 0 and t2["fallback"] == 0 and t2["models"] == {}

print("\nALL ASSERTIONS PASSED")
