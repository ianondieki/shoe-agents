"""An OpenAI-compatible streaming endpoint that lets a voice platform drive the voice agent.

ElevenLabs Agents (or anything else speaking the OpenAI chat-completions protocol) points its
"Custom LLM" at POST /v1/chat/completions. The platform does speech-to-text, turn-taking,
barge-in and text-to-speech; every word of the conversation still comes from build_agent, so
the tools, the price policy, the approval gate and the traces are exactly the ones the terminal
version uses. Nothing about the shop lives here - only what a phone line needs.

Run:        python voice_server.py            (or: uvicorn voice_server:app --port 8013)
Try it:     python tests/test_voice_server.py (no API keys, no voice, no network)

What the platform sends, and what we do with it:
  - The WHOLE transcript every turn. Our agent keeps its own memory per session, so only the
    user messages we have not seen yet are fed in; the platform's copy of what we said is ignored.
  - Its own system prompt and tools. The prompt is ignored (ours in voice_agent.py is the one
    source of truth). Its tools are NOT: end_call arrives there, and the platform only hangs up
    when our reply contains a matching tool call, so our agent has an end_call of its own and the
    server translates it on the way out.
  - Possibly the same request twice, on a timeout. Replays are answered from cache, and the price
    quote state machine makes a repeated order a no-op anyway.
"""
import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from agent_core import build_agent
from db import init_db
from tracing import TRACER
from voice_agent import NAME, SENSITIVE, SYSTEM, TOOLS, voice_approve, voice_candidates

SHARED_SECRET = os.getenv("VOICE_SHARED_SECRET", "")
RECURSION_LIMIT = int(os.getenv("VOICE_RECURSION_LIMIT", "8"))

# Spoken the moment a tool call starts, if the model has not already said something itself.
# Written by hand, not generated: guaranteed to fire, and costs no generation time. Never
# "one moment while I process that" - the caller must not hear the machinery.
FILLER = {
    "customer_and_stock": "Let me have a look.",
    "evaluate_offer": "Let me see what I can do on that.",
    "place_order": "Putting that through now.",
    "send_email": "Sending that over now.",
    # end_call ends the turn, so the model gets no second chance to speak: if it hung up without a
    # word, this is its goodbye.
    "end_call": "Thanks for calling, goodbye!",
}
FALLBACK_REPLY = "Sorry, I lost my place there. Which pair were we talking about?"


@tool
def end_call(reason: str = "") -> str:
    """Hang up the phone. Only after the customer has said goodbye or has nothing more to ask."""
    return "The line will close after this reply. Say a short, warm goodbye and nothing else."


CALL_CONTROL = """

When the customer says goodbye - bye, that's all, that's everything - answer anything they just
asked in one short sentence, say goodbye, and call end_call in that same reply. Finishing an order is
not a goodbye: after an order goes through, tell them it is done and ask if there is anything else.
Never hang up in the middle of a haggle."""

# Backstop for a model that says goodbye but forgets the tool: when the caller AND the reply both
# say goodbye, the line is closed anyway. Both sides are required, so a stray "bye" never ends a
# call, and the order-turn guard still applies on top.
_CALLER_BYE = re.compile(r"\b(good ?bye|bye|that'?s all|that is all|that'?s everything)\b", re.I)
_REPLY_BYE = re.compile(r"\b(good ?bye|bye|take care|have a (great|good|nice) (day|one|run))\b", re.I)

SERVER_TOOLS = [*TOOLS, end_call]


def build_server_agent(llm=None):
    """The voice agent plus call control. Kept separate from voice_agent.py on purpose: hanging
    up is a property of the phone line, not of the shop, and the terminal agent has no line."""
    init_db()
    return build_agent(
        NAME, SYSTEM + CALL_CONTROL, SERVER_TOOLS, SENSITIVE, voice_approve,
        llm if llm is not None else voice_candidates(SERVER_TOOLS),
        recursion_limit=RECURSION_LIMIT,
        # Hanging up ends the turn. Looping back to the model after end_call made it say its whole
        # goodbye a second time in a live simulation - and cost a model call to do it.
        terminal_tools=frozenset({"end_call"}),
    )


# ---------- making text safe to read aloud ----------
# The prompt asks for spoken English and the model mostly complies, but "mostly" is audible: in a
# live run it wrote "$113" (read as "dollar one hundred thirteen"), narrated "*ends call*" as a
# stage direction, and ran "$120.Great" together. Same principle as the price floor - do not rely
# on the model for something code can guarantee.
_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*")               # **Trail runner**: keep the words
# A *stage direction* is one that stands as its own clause - at the start, or straight after a
# sentence ends - and is dropped. Asterisks mid-sentence are emphasis and keep their word: getting
# this wrong would turn "that's *not* the price" into "that's the price".
_STAGE = re.compile(r"(^|[.!?]\s+)\*[^*\n]{1,60}\*\s*")
_EMPH = re.compile(r"\*([^*\n]+?)\*")
_MONEY = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{1,2}))?")
_MARKUP = re.compile(r"[*_#`~|>]+")                     # stray markdown nobody should hear
# "$120.Great" -> "120 dollars. Great"; "dollars.120" -> "dollars. 120". A letter must precede the
# stop for a digit to follow it, so a decimal like 119.99 is never split.
_RUN_ON = re.compile(r"([.!?])([A-Z])|(?<=[A-Za-z])([.!?])(\d)")
_SPACES = re.compile(r"[ \t\r\n]+")


def _dollars(m: re.Match) -> str:
    whole = m.group(1).replace(",", "")
    cents = int((m.group(2) or "0").ljust(2, "0"))
    return f"{whole} dollars" + (f" and {cents} cents" if cents else "")


def speakable(text: str) -> str:
    text = _BOLD.sub(r"\1", text)
    text = _STAGE.sub(r"\1", text)
    text = _EMPH.sub(r"\1", text)
    text = _MONEY.sub(_dollars, text)
    text = _MARKUP.sub("", text)
    text = _RUN_ON.sub(lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", text)
    return _SPACES.sub(" ", text)


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _drop_repeats(piece: str, said: set[str]) -> str:
    """Never say the same sentence twice in one reply. A model retrying a refused tool restates itself
    before each attempt - live, "What price were you thinking?" came out three times in a row."""
    core = piece.strip()
    if not core:
        return piece
    kept = []
    for sentence in _SENTENCE.split(core):
        key = re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()
        if key and key in said:
            continue
        said.add(key)
        kept.append(sentence)
    if not kept:
        return ""
    lead, trail = piece[:len(piece) - len(piece.lstrip())], piece[len(piece.rstrip()):]
    return lead + " ".join(kept) + trail


def _cut(buf: str) -> int:
    """How much of the buffer is ready to speak: through the last finished sentence, or through the
    last word once it is long. Whole sentences are what text-to-speech phrases naturally, and a
    sanitiser needs whole tokens - "$" and "113" can arrive in different chunks."""
    end = max(buf.rfind(p) for p in (". ", "! ", "? "))
    if end >= 0:
        return end + 2
    if len(buf) > 80:
        # Cut at a word, but never just before an asterisk: the next chunk would then start with
        # "*word*", which the sanitiser would mistake for a sentence-opening stage direction.
        i = len(buf)
        while (i := buf.rfind(" ", 0, i)) >= 0:
            if buf[i + 1:i + 2] != "*":
                return i + 1
    return 0


# ---------- protocol helpers ----------
def _text(content) -> str:
    """OpenAI message content is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict)).strip()
    return ""


def _session_id(body: dict, request: Request) -> str:
    """Which conversation this is. The field ElevenLabs uses is unconfirmed, so try the plausible
    ones and log the keys of the first request per session to find out for certain."""
    extra = body.get("elevenlabs_extra_body") or {}
    return str(
        extra.get("conversation_id") or extra.get("system__conversation_id")
        or body.get("user") or request.headers.get("x-session-id") or "default"
    )


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> str:
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _end_call_delta() -> dict:
    return {"tool_calls": [{"index": 0, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                            "function": {"name": "end_call", "arguments": "{}"}}]}


# ---------- the app ----------
def create_app(agent=None) -> FastAPI:
    """A factory so tests can pass a scripted agent and get fully isolated session state."""
    state = {"agent": agent}
    sessions: dict[str, dict] = {}       # session id -> {"heard": [...], "epoch": n, "reply": str, "hung_up": bool}
    locks: dict[str, asyncio.Lock] = {}  # one turn at a time per call
    tracer_lock = asyncio.Lock()         # TRACER holds one in-flight turn, so turns are serialised

    @asynccontextmanager
    async def lifespan(_app):
        if state["agent"] is None:
            # Build at startup, never on the first request: the first caller would otherwise sit in
            # silence through model client construction and a multi-second provider SDK import.
            t0 = time.perf_counter()
            state["agent"] = build_server_agent()
            print(f"  🔥 voice agent warm in {int((time.perf_counter() - t0) * 1000)}ms")
        yield

    app = FastAPI(title="shoe-agent voice endpoint", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": state["agent"] is not None, "sessions": len(sessions)}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if SHARED_SECRET and request.headers.get("authorization") != f"Bearer {SHARED_SECRET}":
            raise HTTPException(status_code=401, detail="bad or missing bearer token")
        if state["agent"] is None:
            raise HTTPException(status_code=503, detail="agent still warming up")

        body = await request.json()
        sid = _session_id(body, request)
        model = str(body.get("model") or NAME)
        stream = bool(body.get("stream", True))
        session = sessions.setdefault(sid, {"heard": [], "epoch": 0, "reply": "", "hung_up": False})
        users = [_text(m.get("content")) for m in body.get("messages", []) if m.get("role") == "user"]

        # A platform that sends no conversation id makes every call look like the same session.
        # A transcript that is shorter than what we have heard, or opens differently, is therefore
        # a NEW call, not a retry: give it a fresh thread instead of the last call's cached reply.
        heard = session["heard"]
        if len(users) < len(heard) or (heard and users and users[0] != heard[0]):
            session.update(heard=[], epoch=session["epoch"] + 1, reply="", hung_up=False)
            heard = session["heard"]
        if not heard and not session["reply"]:
            tool_names = [(t.get("function") or {}).get("name") for t in body.get("tools") or []]
            print(f"  📞 new call in session {sid!r} (epoch {session['epoch']}); "
                  f"x-session-id header: {request.headers.get('x-session-id')!r}; "
                  f"headers: {sorted(h for h in request.headers.keys() if h != 'authorization')}; "
                  f"request keys: {sorted(body)}; "
                  f"extra-body keys: {sorted((body.get('elevenlabs_extra_body') or {}))}; "
                  f"platform tools: {tool_names}")
        fresh = [u for u in users[len(heard):] if u]
        messages = body.get("messages") or []
        last_role = messages[-1].get("role") if messages else None

        queue: asyncio.Queue = asyncio.Queue()
        if fresh:
            # Several unanswered utterances (the caller spoke twice, or talked over us) become one turn.
            text = " ".join(fresh)
            # Mark these words as heard NOW, not when the turn ends: a duplicate of this request that
            # arrives mid-turn must join this turn, not start a second one. (Live, a slow first turn
            # was sent twice; both ran, and Mo greeted the caller with "Still here, John?")
            session["heard"] = list(users)
            done = asyncio.Event()
            session["inflight"] = done
            lock = locks.setdefault(sid, asyncio.Lock())
            # The turn runs as its own task so it always finishes: if the caller barges in the platform
            # closes our stream, but an order half-way through place_order must still commit and land
            # in this session's history. The next turn simply waits on the lock.
            asyncio.create_task(_run_turn(state["agent"], sid, text, list(users), session, lock, queue, done))
        elif last_role != "user":
            # Nothing new from the caller, and the transcript ends on OUR last message: the platform
            # ran end_call and is asking what we do next. Observed live, three wrong answers:
            #   - replay the goodbye + end_call: the goodbye was spoken three times;
            #   - an empty reply: ElevenLabs' simulator crashed with a 500;
            # so the answer is the truthful one - hang up again, with no words. Nothing audible can
            # repeat, and the reply is never empty.
            tail = [(m.get("role"), bool(m.get("tool_calls")), (_text(m.get("content")) or "")[:30])
                    for m in messages[-3:]]
            print(f"  ↩️  {sid!r}: continuation (last role {last_role!r}) -> end_call, no words; tail={tail}")
            await queue.put(("done", {"hung_up": session["hung_up"], "continuation": True}))
        else:
            # The same request again (a timeout retry). If the original is still running, wait for it
            # and give the same answer, rather than answering twice.
            inflight = session.get("inflight")
            if inflight is not None and not inflight.is_set():
                await inflight.wait()
            print(f"  🔁 {sid!r}: retry of an answered turn - replaying")
            await queue.put(("text", session["reply"] or "Sorry, could you say that again?"))
            await queue.put(("done", {"hung_up": session["hung_up"], "replay": True}))

        if not stream:
            spoken, meta = [], {}
            while (item := await queue.get())[0] != "done":
                spoken.append(item[1])
            meta = item[1]
            message = {"role": "assistant", "content": "".join(spoken)}
            if meta.get("hung_up"):
                message["tool_calls"] = _end_call_delta()["tool_calls"]
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if meta.get("hung_up") else "stop"}],
            })

        async def sse():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            yield _chunk(cid, model, {"role": "assistant", "content": ""})
            while True:
                kind, payload = await queue.get()
                if kind == "text":
                    yield _chunk(cid, model, {"content": payload})
                elif kind == "done":
                    if payload.get("hung_up"):
                        yield _chunk(cid, model, _end_call_delta())
                        yield _chunk(cid, model, {}, "tool_calls")
                    else:
                        yield _chunk(cid, model, {}, "stop")
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _run_turn(agent, sid, text, users, session, lock, queue, done):
        async with lock, tracer_lock:
            t0 = time.perf_counter()
            turn_id = TRACER.begin_turn(text)
            # `utterance` reaches the tools: place_order reads it to tell a yes from a counter-offer.
            cfg = {"configurable": {"thread_id": f"voice:{sid}:{session['epoch']}", "turn_id": turn_id,
                                    "depth": 0, "utterance": text},
                   "recursion_limit": RECURSION_LIMIT}
            spoken, fillers, calls = [], [], set()
            ttft, hung_up, ordered, error = None, False, False, None
            last_msg_id, segment_spoke, boundary = None, False, False
            pending = ""  # model text not yet safe to speak (an unfinished sentence)

            said: set[str] = set()

            async def emit(piece: str):
                nonlocal ttft
                piece = _drop_repeats(piece, said)
                if not piece.strip():
                    return
                if ttft is None:
                    ttft = int((time.perf_counter() - t0) * 1000)
                spoken.append(piece)
                await queue.put(("text", piece))

            async def push(piece: str, final: bool = False):
                """Buffer model text and release it a finished sentence at a time, sanitised."""
                nonlocal pending
                pending += piece
                cut = len(pending) if final else _cut(pending)
                if cut:
                    out, pending = pending[:cut], pending[cut:]
                    await emit(speakable(out))

            async def new_thing_to_say():
                """Finish the last thing, and make sure the next word does not run into it."""
                nonlocal pending
                await push("", final=True)
                if spoken and not spoken[-1].endswith(" "):
                    pending = " "

            try:
                async for msg, meta in agent.astream({"messages": [HumanMessage(text)]}, cfg,
                                                     stream_mode="messages"):
                    if meta.get("langgraph_node") != "agent":
                        # Tool results are data for the model, never words for the caller - but a
                        # tool running means whatever the model says next is a separate utterance.
                        boundary = True
                        continue
                    # Three signals that a new model message began, because no single one is
                    # reliable across providers: a tool ran, a tool was called, or the id changed.
                    if boundary or (msg.id and msg.id != last_msg_id):
                        if segment_spoke:
                            await new_thing_to_say()
                        last_msg_id, segment_spoke, boundary = msg.id, False, False
                    piece = _text(msg.content)
                    if piece:
                        if not segment_spoke and spoken and not pending and not spoken[-1].endswith(" "):
                            pending = " "
                        await push(piece)
                        segment_spoke = True
                    # A streaming model announces a call in tool_call_chunks as it is generated;
                    # a non-streaming one delivers the finished message with tool_calls. Both occur
                    # (a fallback candidate may not stream), and missing either would silently skip
                    # the filler and, worse, never hang up.
                    calls_here = (getattr(msg, "tool_call_chunks", None)
                                  or getattr(msg, "tool_calls", None) or [])
                    for tc in calls_here:
                        name = tc.get("name")
                        key = tc.get("id") or f"{msg.id}:{tc.get('index')}"
                        if not name or key in calls:
                            continue
                        calls.add(key)
                        boundary = True  # the model must wait for the tool before speaking again
                        if name == "place_order":
                            ordered = True
                        if name == "end_call":
                            hung_up = True
                        if not segment_spoke and name in FILLER:
                            # Spoken at once, never held for a sentence end: its whole job is to
                            # fill the silence while the tool runs.
                            await new_thing_to_say()
                            await push(FILLER[name] + " ", final=True)
                            fillers.append(name)
                            segment_spoke = True
                await push("", final=True)
            except GraphRecursionError:
                error = "recursion limit"
                await new_thing_to_say()
                await push(FALLBACK_REPLY, final=True)
            except Exception as e:  # every model down, or a bug: never leave the caller in silence
                error = f"{type(e).__name__}: {e}"
                await new_thing_to_say()
                await push(FALLBACK_REPLY, final=True)
            finally:
                # Never hang up on the turn that placed an order. A live simulation did exactly this:
                # "deal, email me the confirmation" -> order, email AND end_call in one reply, so the
                # caller never got to hear it was done, ask a question, or say goodbye.
                reply = "".join(spoken).strip()
                if not hung_up and not error and _CALLER_BYE.search(text) and _REPLY_BYE.search(reply):
                    hung_up = True
                    TRACER.event("voice_guard", session=sid, turn_id=turn_id,
                                 note="end_call added: caller and reply both said goodbye")
                # An order turn only ends the call if the caller said goodbye themselves ("deal -
                # bye!"): they have heard it is done. Otherwise they have not had a chance to react.
                if hung_up and ordered and not _CALLER_BYE.search(text):
                    hung_up = False
                    TRACER.event("voice_guard", session=sid, turn_id=turn_id,
                                 note="end_call suppressed: an order was placed and the caller had not said goodbye")
                session.update(heard=users, reply=reply, hung_up=hung_up)
                TRACER.event("voice", session=sid, turn_id=turn_id, ttft_ms=ttft,
                             ms=int((time.perf_counter() - t0) * 1000), fillers=fillers,
                             hung_up=hung_up, reply_chars=len(reply), error=error)
                TRACER.end_turn(error)
                done.set()  # any duplicate request waiting on this turn can now replay its answer
                await queue.put(("done", {"hung_up": hung_up}))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("VOICE_PORT", "8013"))
    if not SHARED_SECRET:
        print("  ⚠️  VOICE_SHARED_SECRET is not set - anyone who finds this URL can talk to the shop.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
