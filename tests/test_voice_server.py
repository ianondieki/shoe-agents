"""The voice endpoint, driven exactly as a voice platform would drive it - with no voice.

A scripted model that really streams (token chunks, and tool calls as tool_call_chunks) runs
behind the real FastAPI app over an in-process transport: no API keys, no network, no quota.
Everything the platform relies on is asserted from the raw SSE bytes it would receive.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import asyncio
import json
import os
import sys
from typing import Any, Iterator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
os.makedirs(SCRATCH, exist_ok=True)
DB = os.path.join(SCRATCH, "voice_server.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
# Isolated from the real .env: no shared secret (test 10 sets its own) and no real email.
os.environ["VOICE_SHARED_SECRET"] = ""
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "voice_server_trace.jsonl")
if os.path.exists(os.environ["TRACE_PATH"]):
    os.remove(os.environ["TRACE_PATH"])
sys.path.insert(0, ROOT)

import httpx  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

import voice_server  # noqa: E402


class Scripted(BaseChatModel):
    """Replies from a script, streaming the way a real provider does: words as content chunks,
    then each tool call as a tool_call_chunk. Records what it was shown, so tests can check that
    the server fed only the new utterance rather than the platform's whole transcript."""
    script: list = []
    seen: list = []
    delay: float = 0.0  # seconds per model call, to make a turn slow enough to be retried mid-flight

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _next(self, messages) -> AIMessage:
        if self.delay:
            import time
            time.sleep(self.delay)
        self.seen.append([getattr(m, "content", "") for m in messages if m.type == "human"])
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        msg = self._next(messages)
        words = msg.content.split(" ") if msg.content else []
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            if run_manager:
                run_manager.on_llm_new_token(piece)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        for i, tc in enumerate(msg.tool_calls):
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                "name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata={
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110}))


def ai(text="", calls=()):
    return AIMessage(content=text, tool_calls=[
        {"name": n, "args": a, "id": f"call_{n}_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)])


def parse_sse(raw: str) -> dict:
    """Decode an SSE body the way a client would, checking the protocol as we go."""
    events = [line[len("data: "):] for line in raw.split("\n") if line.startswith("data: ")]
    assert events and events[-1] == "[DONE]", f"stream must end with [DONE]: {events[-3:]}"
    chunks = [json.loads(e) for e in events[:-1]]
    for c in chunks:
        assert c["object"] == "chat.completion.chunk" and len(c["choices"]) == 1, c
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk must carry the role"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    tool_calls = [tc for c in chunks for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0]["finish_reason"]]
    assert len(finish) == 1, f"exactly one finish_reason expected, got {finish}"
    pieces = [c["choices"][0]["delta"]["content"] for c in chunks if c["choices"][0]["delta"].get("content")]
    return {"text": text, "tool_calls": tool_calls, "finish": finish[0], "n_chunks": len(chunks),
            "pieces": pieces}


async def post(client, messages, session="call-A", stream=True, headers=None):
    body = {"model": "shoe", "stream": stream, "messages": messages,
            "elevenlabs_extra_body": {"conversation_id": session}}
    r = await client.post("/v1/chat/completions", json=body, headers=headers or {})
    return r


async def main():
    real_outbox_before = _safety.real_outbox_snapshot()
    model = Scripted(script=[], seen=[])
    app = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # ---- 1. a plain spoken reply is valid OpenAI SSE, streamed in pieces ----
        model.script = [ai("Hey Jane, the road runner is ninety dollars. Want to try it?")]
        convo = [{"role": "system", "content": "the platform's own prompt, which we ignore"},
                 {"role": "assistant", "content": "Thanks for calling, who am I speaking with?"},
                 {"role": "user", "content": "Hi, it's Jane Doe."}]
        r = await post(client, convo)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), r
        out = parse_sse(r.text)
        assert out["text"] == "Hey Jane, the road runner is ninety dollars. Want to try it?", out["text"]
        assert out["finish"] == "stop" and not out["tool_calls"]
        # Streamed a finished sentence at a time: the first ships while the second is still being
        # written, which is the latency win, and each piece is whole so it can be sanitised.
        assert out["pieces"] == ["Hey Jane, the road runner is ninety dollars. ", "Want to try it?"], out["pieces"]
        print(f"1. plain reply: sent as {len(out['pieces'])} sentences, role first, [DONE] last, finish=stop")

        # ---- 2. only the NEW utterance reaches the agent, not the platform's whole transcript ----
        model.script = [ai("Sure, the trail runner is one twenty.")]
        convo += [{"role": "assistant", "content": out["text"]},
                  {"role": "user", "content": "How much is the trail runner?"}]
        out = parse_sse((await post(client, convo)).text)
        last_turn_humans = model.seen[-1]
        assert last_turn_humans == ["Hi, it's Jane Doe.", "How much is the trail runner?"], last_turn_humans
        print(f"2. second turn: agent history holds each utterance once: {last_turn_humans}")

        # ---- 3. a retried request is answered from cache, not re-run ----
        calls_before = len(model.seen)
        replay = parse_sse((await post(client, convo)).text)
        assert replay["text"] == out["text"] and len(model.seen) == calls_before, "retry re-ran the agent"
        print("3. identical retry: same reply, agent not invoked again")

        # ---- 4. a tool call with no words first gets a hand-written filler, then the answer ----
        model.script = [
            ai(calls=[("customer_and_stock", {"customer_name": "Jane Doe", "activity": "running"})]),
            ai("The road runner's ninety and the trail runner's one twenty."),
        ]
        convo += [{"role": "assistant", "content": out["text"]},
                  {"role": "user", "content": "What running shoes have you got?"}]
        out = parse_sse((await post(client, convo)).text)
        assert out["text"].startswith("Let me have a look. "), out["text"]
        assert out["text"].endswith("trail runner's one twenty."), out["text"]
        print(f"4. tool turn: {out['text']!r}")

        # ---- 5. when the model already spoke before its tool call, no filler is added ----
        model.script = [
            ai("Let me see what I can do on that.",
               calls=[("evaluate_offer", {"shoe_id": 102, "customer_id": 1, "customer_offer": 80})]),
            ai("Best I can do is one oh eight."),
        ]
        convo += [{"role": "assistant", "content": out["text"]},
                  {"role": "user", "content": "I'll give you eighty for the trail runner."}]
        out = parse_sse((await post(client, convo)).text)
        assert out["text"].count("Let me see what I can do") == 1, out["text"]
        print(f"5. model's own filler kept, ours suppressed: {out['text']!r}")

        # ---- 6. end_call becomes the tool call the platform hangs up on - and ends the turn ----
        # A second scripted reply is queued on purpose: if the graph looped back to the model after
        # end_call (the bug that made Mo say his goodbye twice), it would be spoken and fail this.
        model.script = [ai("Thanks Jane, enjoy the run!", calls=[("end_call", {"reason": "goodbye"})]),
                        ai("Thanks Jane, enjoy the run!")]
        convo += [{"role": "assistant", "content": out["text"]},
                  {"role": "user", "content": "That's all, thanks, bye!"}]
        calls_before = len(model.seen)
        out = parse_sse((await post(client, convo)).text)
        assert out["finish"] == "tool_calls", out
        assert [tc["function"]["name"] for tc in out["tool_calls"]] == ["end_call"], out["tool_calls"]
        assert out["text"].strip() == "Thanks Jane, enjoy the run!", f"goodbye repeated or lost: {out['text']!r}"
        assert len(model.seen) == calls_before + 1, "the model was called again after end_call"
        model.script.clear()
        print(f"6. goodbye: spoke {out['text']!r} once, emitted end_call, and the model was not called again")

        # ---- 6b. a model that hangs up without a word still says goodbye (the end_call filler) ----
        model.script = [ai(calls=[("end_call", {})])]
        out = parse_sse((await post(client, [{"role": "user", "content": "Bye!"}], session="silent-bye")).text)
        assert out["text"].strip() == voice_server.FILLER["end_call"] and out["finish"] == "tool_calls", out
        print(f"6b. silent hang-up gets a spoken goodbye: {out['text']!r}")

        # ---- 7. sessions are isolated: a second caller starts from nothing ----
        model.script = [ai("Hello! Who am I speaking with?")]
        out = parse_sse((await post(client, [{"role": "user", "content": "Hello?"}], session="call-B")).text)
        assert model.seen[-1] == ["Hello?"], model.seen[-1]
        print("7. second session: sees only its own words")

        # ---- 8. non-streaming mode, for curl and for debugging ----
        model.script = [ai("Still here.")]
        r = await post(client, [{"role": "user", "content": "You there?"}], session="call-C", stream=False)
        j = r.json()
        assert j["object"] == "chat.completion" and j["choices"][0]["message"]["content"] == "Still here.", j
        print("8. stream=false returns one ordinary chat.completion")

        # ---- 9. a crashing model never leaves the caller in silence ----
        model.script = []  # pop from an empty script -> IndexError inside the graph
        out = parse_sse((await post(client, [{"role": "user", "content": "Hello?"}], session="call-D")).text)
        assert out["text"] == voice_server.FALLBACK_REPLY and out["finish"] == "stop", out
        print(f"9. model failure: caller hears {out['text']!r}")

    # ---- 10. with a shared secret configured, an unauthenticated request is refused ----
    voice_server.SHARED_SECRET = "s3cret"
    try:
        app2 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://test") as c2:
            assert (await post(c2, [{"role": "user", "content": "hi"}])).status_code == 401
            model.script = [ai("Hi.")]
            ok = await post(c2, [{"role": "user", "content": "hi"}], headers={"authorization": "Bearer s3cret"})
            assert ok.status_code == 200, ok.status_code
    finally:
        voice_server.SHARED_SECRET = ""
    print("10. shared secret: missing token -> 401, correct token -> 200")

    # ---- 12. the sanitiser, on the exact strings a live run produced ----
    sp = voice_server.speakable
    assert sp("it's only $90.") == "it's only 90 dollars.", sp("it's only $90.")
    assert sp("That's $120.Great, that's $120.") == "That's 120 dollars. Great, that's 120 dollars."
    assert "*" not in sp("*ends call* Goodbye! Bye now!") and "ends call" not in sp("*ends call* Goodbye!")
    assert sp("$1,200.50 total") == "1200 dollars and 50 cents total", sp("$1,200.50 total")
    assert sp("the **Trail** runner") == "the Trail runner", sp("the **Trail** runner")
    # emphasis mid-sentence keeps its word - dropping it would invert the meaning
    assert sp("that's *not* the price") == "that's not the price", sp("that's *not* the price")
    assert sp("Sure. *laughs* Okay then.") == "Sure. Okay then.", sp("Sure. *laughs* Okay then.")
    # a terminal test call: the shop's own shoe number read out, and a dash the cutter split after
    live = "Hey Jane, I’ve got a great pair for you—Shoe 101, the lightweight road runner in black."
    assert sp(live) == "Hey Jane, I’ve got a great pair for you, the lightweight road runner in black.", sp(live)
    assert sp("Shoe 103 is waterproof.") == "That pair is waterproof.", sp("Shoe 103 is waterproof.")
    assert sp("The trail runner (ID: 102) is 120 dollars.") == "The trail runner is 120 dollars."
    assert sp("I can do the trail shoe 108 dollars.") == "I can do the trail shoe 108 dollars.", "a price is not an id"
    assert sp("It comes in size 42 and 43.") == "It comes in size 42 and 43."
    cut = voice_server._cut
    # a sentence of ordinary length is spoken whole: cut early, "...perfect for your | runs." was heard
    assert cut(live[:-1]) == 0 and cut("Hi Jane, I've got a lightweight road runner in black that's perfect for your") == 0
    # a long one splits at its last pause, so it is still phrased like speech
    long = ("Hey Jane, I've got a great pair for you, the lightweight road runner in black, and it is super "
            "comfy on long runs and on the treadmill as well")
    assert long[:cut(long)].endswith("in black, "), long[:cut(long)]
    # review: an id is only taken out if it is the shop's; prices, sizes and order numbers stay whole
    for said, heard in (("I can do this shoe 108.", "I can do this shoe 108."),
                        ("the item number 105 in beige", "the pair in beige"),
                        ("I'd go with shoe 101 for trails.", "I'd go with that pair for trails."),
                        ("Your order ID: 12 is confirmed.", "Your order ID: 12 is confirmed."),
                        ("Thanks, customer ID 1.", "Thanks."),
                        ("Between $120 - $130 for most pairs.", "Between 120 dollars to 130 dollars for most pairs."),
                        ("Call 555-1234 for help.", "Call 555-1234 for help.")):
        assert sp(said) == heard, (said, sp(said))
    # a terminal test call: "Bye. (endcall)" - a bracketed stage direction is not said; a bracketed size is
    assert sp("Bye. (endcall)") == "Bye.", sp("Bye. (endcall)")
    assert sp("Sure (laughs), that's fine.") == "Sure, that's fine.", sp("Sure (laughs), that's fine.")
    assert sp("The black one (size 9) is in stock.") == "The black one (size 9) is in stock."
    # ...and a piece never starts with one - alone, "Shoe 101 and..." would read as a new sentence
    dash = ("Hey Jane, I've got a great pair for you, and I think you will love it on the trails this summer"
            "—Shoe 101 and the lightweight road runner")
    assert cut(dash) and not voice_server._IDS.match(dash, cut(dash)), dash[cut(dash):]
    print("12. sanitiser: $ -> dollars, stage directions dropped, emphasis kept, run-ons split, markdown stripped, "
          "shop ids never read out, long sentences split at a pause")

    # ---- 13. the same defects end to end, through a real streamed turn ----
    model.script = [
        ai("That's $120.", calls=[("evaluate_offer", {"shoe_id": 102, "customer_id": 1, "customer_offer": 120})]),
        ai("Great, that's $120. Shall I lock that in?"),
    ]
    app3 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app3), base_url="http://test") as c3:
        out = parse_sse((await post(c3, [{"role": "user", "content": "How much?"}], session="live-2")).text)
        assert "$" not in out["text"] and "120.Great" not in out["text"], out["text"]
        assert out["text"] == "That's 120 dollars. Great, that's 120 dollars. Shall I lock that in?", out["text"]
        model.script = [ai("*ends call* Goodbye! Bye now!", calls=[("end_call", {})])]
        out = parse_sse((await post(c3, [{"role": "user", "content": "How much?"},
                                         {"role": "user", "content": "Bye!"}], session="live-2")).text)
        assert "*" not in out["text"] and "ends call" not in out["text"], out["text"]
        assert out["finish"] == "tool_calls", out
    print(f"13. live defects replayed: {out['text']!r}, and the line still closes")

    # ---- 14. two calls with NO conversation id must not be mistaken for a retry ----
    app4 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app4), base_url="http://test") as c4:
        async def bare(messages):  # exactly what a platform sends when it identifies nothing
            return await c4.post("/v1/chat/completions", json={"model": "shoe", "stream": True, "messages": messages})
        model.script = [ai("Hi Jane!"), ai("The trail runner is one twenty.")]
        first_call = [{"role": "user", "content": "Hi, it's Jane."}]
        parse_sse((await bare(first_call)).text)
        first_call += [{"role": "assistant", "content": "Hi Jane!"}, {"role": "user", "content": "How much is the trail runner?"}]
        parse_sse((await bare(first_call)).text)
        # the line drops; somebody else rings. One message, different opener, same "default" session.
        model.script = [ai("Hello John, what can I get you?")]
        out = parse_sse((await bare([{"role": "user", "content": "Hello, this is John."}])).text)
        assert out["text"] == "Hello John, what can I get you?", f"second call got a stale reply: {out['text']!r}"
        assert model.seen[-1] == ["Hello, this is John."], f"second call inherited the first call's memory: {model.seen[-1]}"
    print("14. no conversation id: a second call gets a fresh thread, not the first call's cache or memory")

    # ---- 15. the goodbye backstop: both sides say goodbye -> the line closes, even without the tool ----
    app5 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app5), base_url="http://test") as c5:
        model.script = [ai("Yes, it's on its way. Thanks for calling, goodbye!")]
        out = parse_sse((await post(c5, [{"role": "user", "content": "Is it coming to my email? Thanks, goodbye."}],
                                    session="bye-1")).text)
        assert out["finish"] == "tool_calls" and out["tool_calls"][0]["function"]["name"] == "end_call", out
        # only the caller says bye: the model is still talking business, so the line stays open
        model.script = [ai("Before you go, did you want the matching socks?")]
        out = parse_sse((await post(c5, [{"role": "user", "content": "Okay, bye."}], session="bye-2")).text)
        assert out["finish"] == "stop", f"hung up while the shop was still talking: {out}"
        # only the reply says bye: the caller never said goodbye, so no hang-up
        model.script = [ai("Have a great day!")]
        out = parse_sse((await post(c5, [{"role": "user", "content": "Is it waterproof?"}], session="bye-3")).text)
        assert out["finish"] == "stop", f"hung up on a caller who never said goodbye: {out}"
    print("15. goodbye backstop: closes only when BOTH sides say goodbye")

    # ---- 16. an order turn: stay on the line unless the CALLER said goodbye ----
    import db as db_module  # the scripted agent really places the order, so use a fresh quote

    app6 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app6), base_url="http://test") as c6:
        for session, closing, expect in (("ord-1", "Deal, email me the confirmation.", "stop"),
                                         ("ord-2", "Deal, I'll take it. Bye!", "tool_calls")):
            model.script = [ai(calls=[("evaluate_offer", {"shoe_id": 101, "customer_id": 1, "customer_offer": 80})]),
                            ai("I can do 81 dollars.")]
            turn1 = [{"role": "user", "content": "Would you take 80 for the road runner?"}]
            parse_sse((await post(c6, turn1, session=session)).text)
            with db_module.db() as conn:
                qid = conn.execute("SELECT MAX(QuoteID) FROM PriceQuote").fetchone()[0]
            model.script = [ai(calls=[("place_order", {"shoe_id": 101, "customer_id": 1, "quote_id": qid})]),
                            ai("Done, the confirmation is on its way. Goodbye!", calls=[("end_call", {})])]
            out = parse_sse((await post(c6, turn1 + [{"role": "user", "content": closing}], session=session)).text)
            assert out["finish"] == expect, f"{closing!r}: expected {expect}, got {out['finish']} - {out['text']!r}"
            model.script.clear()
    print("16. order turn: 'deal, email me' keeps the line open; 'deal, bye!' closes it")

    # ---- 17. after the platform runs end_call it asks us to continue: say nothing, re-send nothing ----
    # Replaying here spoke the goodbye three times in a live ElevenLabs run: each replay re-sent
    # end_call, which triggered another continuation request.
    app7 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app7), base_url="http://test") as c7:
        model.script = [ai("Thanks for calling, goodbye!", calls=[("end_call", {})])]
        convo7 = [{"role": "user", "content": "That's all, bye!"}]
        first = parse_sse((await post(c7, convo7, session="cont-1")).text)
        assert first["finish"] == "tool_calls"
        calls_before = len(model.seen)
        convo7 += [{"role": "assistant", "content": first["text"], "tool_calls": [
                       {"id": "call_x", "type": "function", "function": {"name": "end_call", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "call_x", "content": "call ended"}]
        cont = parse_sse((await post(c7, convo7, session="cont-1")).text)
        # No words (so nothing is spoken twice), but not empty either (an empty reply crashed
        # ElevenLabs' simulator): the truthful answer after end_call is end_call.
        assert cont["text"] == "", f"the goodbye was spoken again: {cont['text']!r}"
        assert cont["finish"] == "tool_calls" and cont["tool_calls"][0]["function"]["name"] == "end_call", cont
        assert len(model.seen) == calls_before, "the agent ran again on a tool continuation"
    print("17. continuation after end_call: no words, end_call again, agent not invoked")

    # ---- 18. a duplicate request arriving mid-turn joins that turn instead of running it twice ----
    app8 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app8), base_url="http://test") as c8:
        model.script = [ai("Hi John, what can I get you?")]
        model.delay = 0.6
        calls_before = len(model.seen)
        convo8 = [{"role": "user", "content": "My name is John Kamau."}]
        r1, r2 = await asyncio.gather(post(c8, convo8, session="dup-1"),
                                      post(c8, convo8, session="dup-1"))
        model.delay = 0.0
        a, b = parse_sse(r1.text), parse_sse(r2.text)
        assert len(model.seen) == calls_before + 1, f"the same utterance ran {len(model.seen) - calls_before} times"
        assert a["text"] == b["text"] == "Hi John, what can I get you?", (a["text"], b["text"])
    print("18. duplicate request mid-turn: one agent run, both requests get the same answer")

    # ---- 19. a model retrying a refused tool does not repeat itself out loud ----
    app9 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app9), base_url="http://test") as c9:
        # exactly the live pattern: the question restated before each attempt at an invented offer
        model.script = [
            ai("What price were you thinking?",
               calls=[("evaluate_offer", {"shoe_id": 102, "customer_id": 1, "customer_offer": 100})]),
            ai("What price were you thinking?",
               calls=[("evaluate_offer", {"shoe_id": 102, "customer_id": 1, "customer_offer": 80})]),
            ai("What price were you thinking?"),
        ]
        out = parse_sse((await post(c9, [{"role": "user", "content": "That's too much for me."}], session="rep-1")).text)
        assert out["text"].count("What price were you thinking?") == 1, out["text"]
        with db_module.db() as conn:
            invented = conn.execute("SELECT COUNT(*) FROM PriceQuote WHERE SessionID LIKE 'voice:rep-1%'").fetchone()[0]
        assert invented == 0, f"an offer the caller never made was quoted ({invented})"
    print(f"19. refused invented offers: said once ({out['text'].strip()!r}), and no quote was written")
    # three terminal test calls: "I can offer it for 108 dollars. 108 dollars." - the price said twice
    said: set = set()
    assert voice_server._drop_repeats("I can offer it for 108 dollars. 108 dollars. ", said) == \
        "I can offer it for 108 dollars. ", said
    assert voice_server._drop_repeats("Is that good?", said) == "Is that good?", "a new short sentence must stay"
    print("19b. a short sentence that only repeats part of an earlier one ('108 dollars.') is not said twice")

    # ---- 20. the first real call: the model hung up on a customer trying to buy ----
    app10 = voice_server.create_app(agent=voice_server.build_server_agent(llm=[("scripted", model)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app10), base_url="http://test") as c10:
        model.script = [ai("I'm sorry, I can't find that one.", calls=[("end_call", {"reason": "goodbye"})])]
        out = parse_sse((await post(c10, [{"role": "user", "content": "Okay, deal. Just go with the initial one."}],
                                    session="real-1")).text)
        assert out["finish"] == "stop" and not out["tool_calls"], f"hung up on a buying customer: {out}"
    print("20. 'Okay, deal' + the model calling end_call: the line stays open - only the caller ends a call")

    # ---- 11. every turn left a voice event with time-to-first-text in the trace ----
    from tools import wait_for_emails
    wait_for_emails()  # confirmation threads write trace events too; let them finish first
    # json.loads on EVERY line: a torn line (two threads writing at once) fails here, not silently
    ev = [json.loads(l) for l in open(os.environ["TRACE_PATH"], encoding="utf-8") if l.strip()]
    voice = [e for e in ev if e["kind"] == "voice"]
    assert len(voice) >= 8 and all(e["ttft_ms"] is not None for e in voice), voice
    assert any(e["hung_up"] for e in voice) and any(e["fillers"] for e in voice)
    print(f"11. traces: {len(voice)} voice events, each with ttft_ms; hang-up and filler recorded")

    wait_for_emails()
    assert _safety.real_outbox_snapshot() == real_outbox_before, "a test wrote into the project's real outbox"
    print("21. the project's real outbox is exactly as it was before the suite ran")

    print("\nALL VOICE SERVER ASSERTIONS PASSED")


asyncio.run(main())
