"""agent_core's safety machinery: the circuit breaker, approval denials, and the history window.

The adversarial review showed every one of these could be deleted with all other tests still green:
switching the breaker off, benching a 429 for ten minutes, or letting a string denial run the tool.
Each test here fails if its mechanism is removed. No API keys, no network.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, "_tmp"), exist_ok=True)
os.environ["TRACE_PATH"] = os.path.join(HERE, "_tmp", "agent_core_trace.jsonl")
if os.path.exists(os.environ["TRACE_PATH"]):
    os.remove(os.environ["TRACE_PATH"])
sys.path.insert(0, ROOT)

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import agent_core  # noqa: E402
from agent_core import build_agent, reset_breakers  # noqa: E402


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def failing(status, calls: list, recover_after=None):
    """A model that raises `status`; if recover_after is set, it answers after that many failures."""
    def run(messages):
        calls.append(time.monotonic())
        if recover_after is not None and len(calls) > recover_after:
            return AIMessage("recovered")
        raise HTTPError(status)
    return RunnableLambda(run)


def healthy(n=20):
    return FakeMessagesListChatModel(responses=[AIMessage(f"answer {i}") for i in range(n)])


def ask(app, text, turn):
    cfg = {"configurable": {"thread_id": "t", "turn_id": turn, "depth": 0}, "recursion_limit": 10}
    return app.invoke({"messages": [HumanMessage(text)]}, cfg)["messages"][-1].content


# ---------- 1. a dead model is tried once, then stepped over ----------
reset_breakers()
dead_calls = []
app = build_agent("a", "sys", [], llm=[("dead-model", failing(403, dead_calls)), ("good", healthy())])
for i in range(3):
    assert ask(app, "hi", f"d{i}").startswith("answer"), "fallback did not answer"
assert len(dead_calls) == 1, f"a 403 model was called {len(dead_calls)} times; the breaker should stop it after 1"
print(f"1. 403: dead model called once across 3 turns, then skipped")

# ---------- 2. a rate limit gets a SHORT bench, and is probed again afterwards ----------
reset_breakers()
agent_core.RATE_LIMIT_TTL_S = 0.4
rl_calls = []
app = build_agent("b", "sys", [], llm=[("busy", failing(429, rl_calls, recover_after=1)), ("good", healthy())])
ask(app, "hi", "r1")                 # busy 429s -> benched briefly, good answers
ask(app, "hi", "r2")                 # still benched: not called
assert len(rl_calls) == 1, f"a benched model was called again inside its backoff: {len(rl_calls)}"
time.sleep(0.5)
assert ask(app, "hi", "r3") == "recovered", "after the backoff the rate-limited model was not retried"
agent_core.RATE_LIMIT_TTL_S = 30.0
print("2. 429: benched for the backoff only, then probed again and used")

# ---------- 3. when every candidate is benched, one is still tried (never fail without trying) ----------
reset_breakers()
agent_core.RATE_LIMIT_TTL_S = 60.0
a_calls, b_calls = [], []
app = build_agent("c", "sys", [], llm=[("m1", failing(429, a_calls, recover_after=1)),
                                       ("m2", failing(429, b_calls, recover_after=1))])
try:
    ask(app, "hi", "p1")
    raise AssertionError("both models failed, the turn should have raised")
except HTTPError:
    pass
assert ask(app, "hi", "p2") == "recovered", "all benched: the turn failed without a real attempt"
assert len(a_calls) + len(b_calls) == 3, (len(a_calls), len(b_calls))
agent_core.RATE_LIMIT_TTL_S = 30.0
print("3. every model benched: the one freed soonest is probed, and the turn succeeds")

# ---------- 4. one agent's broken stand-in model does not bench another agent's ----------
reset_breakers()
broken = []
app_a = build_agent("broken-agent", "sys", [], llm=failing(403, broken))
app_b = build_agent("fine-agent", "sys", [], llm=healthy())
try:
    ask(app_a, "hi", "k1")
except HTTPError:
    pass
assert ask(app_b, "hi", "k2") == "answer 0", "a different agent's failure benched this agent's model"
print("4. single-model agents have separate breakers: A's 403 does not silence B")

# ---------- 5. a string denial is the tool's output, and the tool never runs ----------
reset_breakers()
ran = []


@tool
def sensitive_action(x: int) -> str:
    """Does something that needs approval."""
    ran.append(x)
    return "done"


reason = "No agreed price on record. Put the customer's number through first."
script = FakeMessagesListChatModel(responses=[
    AIMessage("", tool_calls=[{"name": "sensitive_action", "args": {"x": 1}, "id": "c1", "type": "tool_call"}]),
    AIMessage("understood"),
])
app = build_agent("d", "sys", [sensitive_action], sensitive={"sensitive_action"},
                  approve=lambda *a: reason, llm=script)
cfg = {"configurable": {"thread_id": "deny", "turn_id": "s1", "depth": 0}, "recursion_limit": 10}
state = app.invoke({"messages": [HumanMessage("do it")]}, cfg)
tool_msg = next(m for m in state["messages"] if m.type == "tool")
assert ran == [], "the approver refused, but the tool ran anyway"
assert tool_msg.content == reason, f"the model did not receive the refusal reason: {tool_msg.content!r}"
print("5. approver returned a reason: tool did not run, and the model was told why")

# ---------- 6. the history window never trims away the turn in progress ----------
reset_breakers()
seen = []


@tool
def big_lookup() -> str:
    """Returns a large result."""
    return "x" * 2000


def recorder(messages):
    seen.append([m.type for m in messages])
    if len(seen) == 1:
        return AIMessage("", tool_calls=[{"name": "big_lookup", "args": {}, "id": "b1", "type": "tool_call"}])
    return AIMessage("here you go")


agent_core.MAX_HISTORY_TOKENS = 50  # far smaller than the tool result
try:
    app = build_agent("e", "sys", [big_lookup], llm=RunnableLambda(recorder))
    ask(app, "earlier question", "w1")
    seen.clear()
    ask(app, "current question", "w2")
finally:
    agent_core.MAX_HISTORY_TOKENS = 0
assert all("human" in call for call in seen), f"a model call lost the caller's current words: {seen}"
assert "tool" in seen[-1], f"the tool result of this turn was trimmed away: {seen[-1]}"
print(f"6. tiny history budget: every call still saw the current question and its tool result: {seen[-1]}")

# ---------- 7. a terminal tool ends the turn - but only a round made ONLY of terminal tools ----------
reset_breakers()


@tool
def hang_up() -> str:
    """Ends the call."""
    return "closing"


@tool
def lookup() -> str:
    """Looks something up."""
    return "found it"


def scripted(*replies):
    calls = []
    def run(messages):
        calls.append(1)
        return replies[len(calls) - 1]
    return RunnableLambda(run), calls


def tc(name, i):
    return {"name": name, "args": {}, "id": f"{name}{i}", "type": "tool_call"}


only_terminal, n1 = scripted(AIMessage("bye", tool_calls=[tc("hang_up", 1)]), AIMessage("bye AGAIN"))
app = build_agent("f", "sys", [hang_up, lookup], llm=only_terminal, terminal_tools=frozenset({"hang_up"}))
assert ask(app, "bye", "t1") == "closing" and len(n1) == 1, f"model re-called after a terminal tool: {len(n1)}"

mixed, n2 = scripted(AIMessage("", tool_calls=[tc("lookup", 1), tc("hang_up", 2)]), AIMessage("here's what I found"))
app = build_agent("g", "sys", [hang_up, lookup], llm=mixed, terminal_tools=frozenset({"hang_up"}))
assert ask(app, "check and bye", "t2") == "here's what I found" and len(n2) == 2, "a mixed round ended early"
print("7. terminal tool alone ends the turn (1 model call); mixed with another tool, the model still reports back")

print("\nALL AGENT CORE ASSERTIONS PASSED")
