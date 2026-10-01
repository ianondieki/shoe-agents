"""The terminal phone call, end to end, with no network, no microphone and no speakers.

Real parts: call.py's call logic, the in-process line to Mo (streaming transport included), the
webrtcvad endpointer on REAL recorded speech, voice_server with all its guards, the database and the
confirmation email. Faked: the microphone (fed from recordings), speech-to-text (returns the scripted
line for each utterance it is handed), Mo's voice (silence of the right length), the keyboard, and
the language model (a script). Every hand-off between those parts is the real one.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import asyncio
import io
import json
import os
import queue
import sqlite3
import sys
import threading
import time
import wave
from typing import Iterator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
os.makedirs(os.environ["OUTBOX_DIR"], exist_ok=True)
DB = os.path.join(SCRATCH, "call_test.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "call_test_trace.jsonl")
os.environ["VOICE_SHARED_SECRET"] = ""
sys.path.insert(0, ROOT)

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

import call  # noqa: E402
import speech  # noqa: E402
import voice_server  # noqa: E402
from db import db  # noqa: E402

REAL_DB = os.path.join(ROOT, "shoes.db")
REAL_OUTBOX_BEFORE = _safety.real_outbox_snapshot()
REAL_DB_ORDERS = (sqlite3.connect(REAL_DB).execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0]
                  if os.path.exists(REAL_DB) else None)


def fixture(name) -> bytes:
    with wave.open(os.path.join(HERE, "fixtures", name)) as w:
        return w.readframes(w.getnframes())


# A real recording of a caller's line (16 kHz mono, played through a virtual audio cable and recorded
# back): the endpointer has to find it, for real, on every turn.
SPEECH = fixture("caller_line.wav")


# ---------------------------------------------------------------- fakes
class Scripted(BaseChatModel):
    """Streams scripted replies the way a provider does: content chunks, then tool_call_chunks.
    A script item may be {"msg": ..., "delay": s, "per_word": s} to be slow like a busy model."""
    script: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _next(self):
        item = self.script.pop(0)
        if isinstance(item, dict):
            time.sleep(item.get("delay", 0))
            return item["msg"], item.get("per_word", 0)
        return item, 0

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next()[0])])

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        msg, per_word = self._next()
        for i, word in enumerate((msg.content or "").split(" ")):
            if word:
                if i and per_word:
                    time.sleep(per_word)
                yield ChatGenerationChunk(message=AIMessageChunk(content=word if i == 0 else " " + word))
        for i, tc in enumerate(msg.tool_calls):
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                "name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}]))


def ai(text="", calls=()):
    return AIMessage(content=text, tool_calls=[
        {"name": n, "args": a, "id": f"c_{n}_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)])


def brain(*script):
    model = Scripted(script=list(script))
    return model, voice_server.build_server_agent(llm=[("scripted", model)])


class FakeMic:
    """Behaves like call.Mic: frames go into .q unless muted - which is exactly how echo is refused."""
    def __init__(self):
        self.q, self.level, self.muted = queue.Queue(), 0.0, threading.Event()
        self.muted.set()
        self.echo_offered = self.echo_dropped = 0

    def feed(self, frame: bytes, echo=False):
        self.echo_offered += echo
        if self.muted.is_set():
            self.echo_dropped += echo
            return
        self.q.put((time.monotonic(), frame))      # the capture time, as the real microphone sends it

    def drain(self):
        while not self.q.empty():
            self.q.get_nowait()

    def close(self):
        pass


class FakeEars:
    """Speech-to-text stand-in: the Nth utterance it is handed is the Nth scripted line."""
    def __init__(self, lines):
        self.lines, self.heard = list(lines), []

    async def text(self, pcm: bytes) -> str:
        assert len(pcm) > speech.MIC_RATE * 2 * 0.5, f"handed a suspiciously short utterance: {len(pcm)} bytes"
        said = self.lines.pop(0)
        self.heard.append(said)
        return said

    async def aclose(self):
        pass


class FakeVoice:
    def __init__(self, seconds=0.15):
        self.spoken, self.chars_used, self.seconds = [], 0, seconds

    async def speak(self, text, cache=False):
        self.spoken.append(text)
        return np.zeros(int(speech.VOICE_RATE * self.seconds), dtype=np.int16)

    async def credits_left(self):
        return (0, 10000)

    async def aclose(self):
        pass


class CountingSpeaker(call.Speaker):
    """A silent speaker that remembers each clip it played: True if to the end, False if cut off."""
    def __init__(self):
        super().__init__(silent=True)
        self.played = []

    def play(self, audio, stop):
        finished = super().play(audio, stop)
        self.played.append(finished)
        return finished


class LoudVoice(FakeVoice):
    """Mo with a real voice: speech-shaped audio at a real level, so a room can echo it back."""
    def __init__(self, seconds=1.2):
        super().__init__(seconds)
        # real speech at Mo's rate - gaps between the words and all, which is what a room echoes
        at16 = np.frombuffer(SPEECH, np.int16).astype(np.float32)
        at24 = np.interp(np.linspace(0, len(at16) - 1, int(len(at16) * speech.VOICE_RATE / speech.MIC_RATE)),
                         np.arange(len(at16)), at16)
        want = int(speech.VOICE_RATE * seconds)
        self.clip = np.tile(at24, int(want / len(at24)) + 1)[:want].astype(np.int16)

    async def speak(self, text, cache=False):
        self.spoken.append(text)
        return self.clip


class RoomSpeaker(CountingSpeaker):
    """A silent speaker that remembers what it played and when - so a microphone in the same room
    can hear it come back."""
    def __init__(self):
        super().__init__()
        self.timeline: list[tuple[float, np.ndarray]] = []

    def play(self, audio, stop):
        self.timeline.append((time.monotonic(), np.asarray(audio)))
        return super().play(audio, stop)

    def heard_at(self, when: float, samples: int) -> np.ndarray:
        """`samples` of 16 kHz audio as it would reach a microphone at that moment."""
        out = np.zeros(samples, dtype=np.float32)
        span = int(samples * speech.VOICE_RATE / speech.MIC_RATE)
        for start, audio in self.timeline:
            i = int((when - start) * speech.VOICE_RATE)
            if not (0 <= i < len(audio)):
                continue
            piece = np.asarray(audio[i:i + span], dtype=np.float32)
            if len(piece) < 2:
                continue
            out += np.interp(np.linspace(0, len(piece) - 1, samples), np.arange(len(piece)), piece)
        return out


async def room(the_call, speaker: RoomSpeaker, mic: FakeMic, stop: threading.Event, coupling=0.35,
               caller: bytes | None = None, caller_at=1.0):
    """The microphone in Mo's room: it hears him through the speakers (at `coupling`), the room's own
    hiss, and - from `caller_at` - the caller talking over him."""
    rng = np.random.default_rng(5)
    voice = np.frombuffer(caller, np.int16).astype(np.float32) if caller else None
    t0, step = time.monotonic(), speech.FRAME_SAMPLES
    while not stop.is_set():
        now = time.monotonic()
        frame = coupling * speaker.heard_at(now - 0.06, step) + rng.standard_normal(step) * 25
        if voice is not None and now - t0 >= caller_at:
            i = int((now - t0 - caller_at) * speech.MIC_RATE)
            piece = voice[i:i + step]
            frame[:len(piece)] += piece
        mic.feed(np.clip(frame, -32768, 32767).astype(np.int16).tobytes())
        await asyncio.sleep(speech.FRAME_MS / 1000)


class Keys:
    """The keyboard, scripted: each entry is (condition on the call, key). One key per condition."""
    def __init__(self, the_call, script):
        self.call, self.script = the_call, list(script)

    def __call__(self):
        if self.script and self.script[0][0](self.call):
            return self.script.pop(0)[1]
        return None


def args(**kw):
    a = call.parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def new_call(agent, ears=(), voice=None, mic=None, speaker=None, **kw):
    out = io.StringIO()
    the_call = call.Call(args(**kw), "practice", call.Screen(out=out), line=call.Line(agent),
                         ears=FakeEars(ears), voice=voice or FakeVoice(), mic=mic,
                         speaker=speaker or call.Speaker(silent=True))
    return the_call, out


async def say(mic: FakeMic, pcm: bytes = SPEECH):
    """The caller says one line: a breath of silence, the recording, then silence until it ends."""
    for f in speech.frames_of(bytes(9600) + pcm + bytes(51200)):     # 1.6 s of silence after
        mic.feed(f)
        await asyncio.sleep(0)


HISS = (np.random.default_rng(11).standard_normal(16000 * 2) * 25).astype(np.int16).tobytes()


async def caller(the_call, mic: FakeMic, turns: int, echo=True):
    """Speaks once each time the call starts listening; the room hisses quietly while Mo talks."""
    for _ in range(turns):
        while the_call.state != "listening":
            if echo and the_call.state == "speaking":
                for f in speech.frames_of(HISS):             # the room, while Mo has the floor
                    mic.feed(f, echo=True)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        await say(mic)
        while the_call.state == "listening":
            await asyncio.sleep(0.02)


# ---------------------------------------------------------------- scenarios
async def scenario_full_call():
    model, agent = brain(
        ai(calls=[("customer_and_stock", {"customer_name": "Jane Doe", "activity": "trail"})]),
        ai("Hi Jane! The cushioned trail runner is 120 dollars."))
    lines = ["Hi, it's Jane Doe.", "That's too much. Eighty dollars.", "Okay, deal.", "Thanks, bye!"]
    mic = FakeMic()
    the_call, out = new_call(agent, ears=lines, mic=mic)

    async def brain_script():
        # the model's next turn is queued the moment the caller's line is heard; the order turn
        # needs the quote id that the haggle produced
        seen = 0
        while seen < 3:
            await asyncio.sleep(0.02)
            if len(the_call.ears.heard) > seen:
                seen = len(the_call.ears.heard)
                if seen == 1:
                    model.script += [ai(calls=[("evaluate_offer", {"shoe_id": 102, "customer_id": 1,
                                                                   "customer_offer": 80})]),
                                     ai("Best I can do is 108 dollars.")]
                elif seen == 2:
                    while True:
                        with db() as conn:
                            qid = conn.execute("SELECT MAX(QuoteID) FROM PriceQuote").fetchone()[0]
                        if qid:
                            break
                        await asyncio.sleep(0.02)
                    model.script += [ai(calls=[("place_order", {"shoe_id": 102, "customer_id": 1, "quote_id": qid})]),
                                     ai("Done, your confirmation is on its way. Anything else?")]
                elif seen == 3:
                    model.script += [ai("Thanks Jane, enjoy the run. Goodbye!", calls=[("end_call", {})])]

    feeder = asyncio.create_task(brain_script())
    await asyncio.gather(the_call.run(), caller(the_call, mic, turns=4))
    feeder.cancel()
    await the_call.close()
    the_call.summary()
    return the_call, mic, out.getvalue()


async def scenario_streaming():
    """The first sentence must reach the caller while the model is still working on the rest."""
    _, agent = brain(ai(calls=[("customer_and_stock", {"customer_name": "Jane Doe"})]),
                     {"msg": ai("Hi Jane! The trail runner is 120 dollars."), "delay": 0.8})
    line = call.Line(agent)
    line.mo_said(call.GREETING)
    t0, pieces = time.perf_counter(), []
    async for kind, text in line.turn("Hi, it's Jane Doe."):
        if kind == "say":
            pieces.append((time.perf_counter() - t0, text.strip()))
    total = time.perf_counter() - t0
    await line.aclose()
    return pieces, total


async def scenario_hang_up_with_esc():
    _, agent = brain()
    the_call, _ = new_call(agent, mic=FakeMic())
    original, call.key_pressed = call.key_pressed, Keys(the_call, [(lambda c: c.state == "listening", "\x1b")])
    try:
        await the_call.run()
        raise AssertionError("Esc did not hang up")
    except call.HangUp as h:
        return the_call, h.by
    finally:
        call.key_pressed = original
        await the_call.close()


REPLY = ["Let me tell you all about our shoes.", "First, the road runner.", "Then the trail runner.",
         "Then the mid boot.", "Then the court shoe."]


async def scenario_cut_in():
    _, agent = brain(ai(" ".join(REPLY)))
    mic, speaker = FakeMic(), CountingSpeaker()
    the_call, out = new_call(agent, ears=["Tell me about your shoes."], voice=FakeVoice(0.6), mic=mic,
                             speaker=speaker)
    # Enter, part-way through Mo's second sentence (the greeting and his first are finished)
    keys = Keys(the_call, [(lambda c: c.state == "speaking" and len(speaker.played) >= 2, "\r")])
    original, call.key_pressed = call.key_pressed, keys
    try:
        await the_call.connect()
        await the_call.speak_one(call.GREETING)
        the_call.line.mo_said(call.GREETING)
        task = asyncio.create_task(the_call.hear())
        await caller(the_call, mic, turns=1, echo=False)
        hung = await the_call.respond(await task)
    finally:
        call.key_pressed = original
        await the_call.close()
    synthesised = [t for t in the_call.voice.spoken if t in REPLY]
    return hung, out.getvalue(), speaker.played, synthesised


async def scenario_talk_after_cut_in():
    """Mo is still streaming a long reply when the caller cuts in and starts talking at once."""
    _, agent = brain({"msg": ai("Right. " + " ".join(REPLY)), "per_word": 0.06},
                     ai("Sure, go ahead."))
    mic = FakeMic()
    the_call, out = new_call(agent, ears=["Tell me about your shoes.", "Actually, just the boots."],
                             voice=FakeVoice(0.4), mic=mic)
    keys = Keys(the_call, [(lambda c: c.state == "speaking", "\r")])
    original, call.key_pressed = call.key_pressed, keys
    try:
        await the_call.connect()
        await the_call.speak_one(call.GREETING)
        the_call.line.mo_said(call.GREETING)
        first = asyncio.create_task(the_call.hear())
        await caller(the_call, mic, turns=1, echo=False)
        reply = asyncio.create_task(the_call.respond(await first))
        while not keys.script == [] or not the_call.mic_open:      # cut in, and wait for the mic to open
            await asyncio.sleep(0.01)
        still_streaming = not reply.done()
        await say(mic)                                             # talking while Mo's turn drains
        await reply
        t0 = time.perf_counter()
        second = await the_call.hear()
        waited = time.perf_counter() - t0
    finally:
        call.key_pressed = original
        await the_call.close()
    return still_streaming, second, waited


async def scenario_pause_mid_turn(pause: float, burst: bool = False):
    """"That's a bit steep." ... 0.9 s ... "I'll give you eighty." is ONE turn, not two."""
    _, agent = brain()
    mic = FakeMic()

    class TimedEars(FakeEars):
        async def text(self, pcm):
            await asyncio.sleep(0.1)                     # Whisper takes a moment
            self.heard.append(round(len(pcm) / 32000, 2))
            return f"{len(pcm) / 32000:.2f} seconds of speech"

    the_call, _ = new_call(agent, mic=mic)
    the_call.ears = TimedEars([])
    await the_call.connect()
    turn = asyncio.create_task(the_call.hear())
    while the_call.state != "listening":
        await asyncio.sleep(0.01)
    half = len(SPEECH) // 4 * 2
    frames = list(speech.frames_of(bytes(9600) + SPEECH[:half] + bytes(int(32000 * pause)) + SPEECH[half:]
                                   + bytes(48000)))
    t0 = time.perf_counter()
    for i, f in enumerate(frames):                       # at the pace of a real microphone, by the clock:
        mic.feed(f)                                      # the pause must last 0.9 s, not 0.9 s of sleeps
        if burst:                                        # ...or all at once: a program that fell behind
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(max(0.0, t0 + (i + 1) * speech.FRAME_MS / 1000 - time.perf_counter()))
    text = await turn
    await the_call.close()
    return text, the_call.ears.heard, half / 32000, (len(SPEECH) - half) / 32000


async def scenario_hold_on():
    _, agent = brain({"msg": ai("The trail runner is 120 dollars."), "delay": 1.0})
    the_call, out = new_call(agent)
    the_call.HOLD_ON_AFTER_S = 0.3
    the_call.line.mo_said(call.GREETING)
    await the_call.respond("How much is the trail runner?")
    await the_call.close()
    return the_call.voice.spoken, out.getvalue()


async def scenario_silent_caller():
    _, agent = brain()
    the_call, out = new_call(agent, mic=FakeMic())
    the_call.NUDGE_AFTER_S, the_call.GIVE_UP_AFTER_S = 0.4, 1.2
    t0 = time.perf_counter()
    try:
        await the_call.run()
        raise AssertionError("a silent caller was never given up on")
    except call.HangUp as h:
        by = h.by
    finally:
        await the_call.close()
    return by, the_call.voice.spoken, time.perf_counter() - t0


async def scenario_keyboard():
    _, agent = brain(ai("We have road runners, trail runners, boots and court shoes. What are you after?"),
                     ai("Good choice."))
    voice = FakeVoice(0.5)
    the_call, out = new_call(agent, voice=voice, keyboard=True)
    speaking = lambda c: c.state == "speaking"            # noqa: E731
    typing = lambda c: c.state == "listening"             # noqa: E731
    keys = Keys(the_call, [(speaking, "b"), (speaking, "o")]
                + [(typing, k) for k in "ots\x08s please\r"])
    original, call.key_pressed = call.key_pressed, keys
    try:
        await the_call.connect()
        the_call.line.mo_said(call.GREETING)
        await the_call.respond("What have you got?")         # the caller starts typing over Mo
        typed = await the_call.hear()
        await the_call.respond(typed)
    finally:
        call.key_pressed = original
        await the_call.close()
    return typed, the_call.line.transcript, out.getvalue(), voice.spoken


async def transcriber_keeps_short_answers():
    """The Whisper prompt holds the shop's names and prices. A long stretch of it coming back is Whisper
    echoing its prompt; a short answer is the caller answering - "Jane Doe." is what the prompt is for."""
    vocabulary = speech.shop_vocabulary()
    said = []

    def groq(request):
        return httpx.Response(200, json={"text": said[-1]})

    ears = speech.Transcriber(api_key="test", vocabulary=vocabulary)
    ears.client = httpx.AsyncClient(transport=httpx.MockTransport(groq))
    out = {}
    for text in ("Jane Doe.", "Eighty.", "Eighty dollars.", "One twenty.", "Cushioned trail runner.",
                 vocabulary, vocabulary.split(". ")[1] + "."):
        said.append(text)
        out[text] = await ears.text(SPEECH)
    await ears.aclose()
    return out, vocabulary


async def scenario_enter_while_thinking():
    _, agent = brain({"msg": ai("The trail runner is 120 dollars."), "delay": 0.8})
    speaker = CountingSpeaker()
    the_call, out = new_call(agent, voice=FakeVoice(0.3), speaker=speaker)
    keys = Keys(the_call, [(lambda c: c.state == "thinking", "\r")])       # a habit after speaking
    original, call.key_pressed = call.key_pressed, keys
    try:
        the_call.line.mo_said(call.GREETING)
        await the_call.respond("How much is the trail runner?")
    finally:
        call.key_pressed = original
        await the_call.close()
    return keys.script, speaker.played, out.getvalue()


async def scenario_esc_while_thinking():
    _, agent = brain({"msg": ai("Let me think about that for a long while."), "delay": 3.0})
    the_call, out = new_call(agent)
    keys = Keys(the_call, [(lambda c: c.state == "thinking", "\x1b")])
    original, call.key_pressed = call.key_pressed, keys
    t0 = time.perf_counter()
    try:
        the_call.line.mo_said(call.GREETING)
        await the_call.respond("Can you do ninety?")
        raise AssertionError("Esc did not hang up")
    except call.HangUp as h:
        took, by = time.perf_counter() - t0, h.by
    finally:
        call.key_pressed = original
        await the_call.close()
    return took, by


async def scenario_still_here():
    """Mo: "Are you still there?" Caller: "Yes, I'm here." - the line answers that, not the brain."""
    _, agent = brain()
    mic = FakeMic()
    the_call, out = new_call(agent, ears=["Yes, I'm here.", "Tell me about your boots."], mic=mic)
    the_call.NUDGE_AFTER_S, the_call.GIVE_UP_AFTER_S = 0.4, 30
    await the_call.connect()
    turn = asyncio.create_task(the_call.hear())
    for line in (call.STILL_THERE, call.TAKE_YOUR_TIME):
        while f"Mo:  {line}" not in out.getvalue():          # said on the call (not just cached)
            await asyncio.sleep(0.02)
        while the_call.state != "listening":
            await asyncio.sleep(0.02)
        await say(mic)
    text = await turn
    await the_call.close()
    return text, the_call.voice.spoken, out.getvalue()


class FlakyVoice(FakeVoice):
    """Fails on the sentences it is told to, the way ElevenLabs can: once, or for good."""
    def __init__(self, fail: dict[str, bool]):
        super().__init__(0.1)
        self.fail = fail

    async def speak(self, text, cache=False):
        if text in self.fail:
            raise speech.SpeechUnavailable(f"voice trouble on {text!r}", lasting=self.fail.pop(text))
        return await super().speak(text, cache)


async def scenario_voice_trouble(lasting: bool):
    _, agent = brain(ai("First sentence. Second sentence. Third sentence."))
    voice = FlakyVoice({"First sentence.": lasting})
    speaker = CountingSpeaker()
    the_call, out = new_call(agent, voice=voice, speaker=speaker)
    the_call.line.mo_said(call.GREETING)
    await the_call.respond("Tell me something.")
    await the_call.close()
    return the_call.voice_ok, speaker.played, out.getvalue()


async def scenario_push_to_talk():
    _, agent = brain()
    mic = FakeMic()
    the_call, out = new_call(agent, ears=["I'll give you eighty dollars."], mic=mic, push_to_talk=True)
    fed = []
    keys = Keys(the_call, [(lambda c: c.state == "listening", "\r"), (lambda c: bool(fed), "\r")])
    original, call.key_pressed = call.key_pressed, keys
    try:
        await the_call.connect()
        turn = asyncio.create_task(the_call.hear())
        while keys.script and len(keys.script) == 2:          # wait for the first Enter
            await asyncio.sleep(0.02)
        await say(mic)                                        # nothing is heard until they press Enter...
        fed.append(True)                                      # ...and it ends when they press it again
        text = await asyncio.wait_for(turn, 10)
    finally:
        call.key_pressed = original
        await the_call.close()
    return text


async def scenario_no_voice():
    _, agent = brain(ai("The trail runner is 120 dollars. Want it?"))
    voice = FakeVoice()
    the_call, out = new_call(agent, voice=voice, no_voice=True)
    await the_call.connect()
    the_call.line.mo_said(call.GREETING)
    await the_call.respond("How much is the trail runner?")
    await the_call.close()
    return voice.spoken, out.getvalue()


class DyingSpeaker(CountingSpeaker):
    def play(self, audio, stop):
        raise call.SpeakerBroken("PortAudioError: device unavailable")


async def scenario_speakers_die():
    _, agent = brain(ai("First sentence. Second sentence."))
    the_call, out = new_call(agent, speaker=DyingSpeaker())
    the_call.line.mo_said(call.GREETING)
    hung = await the_call.respond("Tell me something.")
    await the_call.close()
    return hung, the_call.voice_ok, out.getvalue()


async def scenario_talk_over_mo(caller: bytes | None, coupling=0.35):
    """Mo is mid-reply in a room where his own voice comes back into the microphone. Talking over him
    must stop him; his own echo must not."""
    _, agent = brain(ai(" ".join(REPLY)))
    mic, speaker, voice = FakeMic(), RoomSpeaker(), LoudVoice(1.2)
    the_call, out = new_call(agent, ears=["Actually, hold on."], voice=voice, mic=mic, speaker=speaker)
    await the_call.connect()
    stop = threading.Event()
    noise = asyncio.create_task(room(the_call, speaker, mic, stop, coupling=coupling,
                                     caller=caller, caller_at=1.2))
    try:
        the_call.line.mo_said(call.GREETING)
        hung = await the_call.respond("Tell me about your shoes.")
        try:
            heard = await asyncio.wait_for(the_call.hear(), 20) if caller else None
        except asyncio.TimeoutError:
            print("TIMED OUT. screen:\n" + out.getvalue())
            print("ears left:", the_call.ears.lines, "heard:", the_call.ears.heard,
                  "carry:", len(the_call.carry), "queue:", the_call.mic.q.qsize(),
                  "floor:", the_call.ear.floor, "coupling:", the_call.guard.coupling)
            raise
    finally:
        stop.set()
        noise.cancel()
        await asyncio.gather(noise, return_exceptions=True)
        await the_call.close()
    return hung, speaker.played, out.getvalue(), heard, the_call.guard.coupling


async def scenario_own_voice_back():
    """A room that fools the guard: Mo stops for his own voice, and it comes back as words. Those must
    never reach his brain - he would be answering himself."""
    _, agent = brain(ai("First sentence. Second sentence. Third sentence."), ai("Of course."))
    mic, speaker, voice = FakeMic(), RoomSpeaker(), LoudVoice(1.2)
    the_call, out = new_call(agent, ears=["Second sentence.", "Actually, hold on."], voice=voice,
                             mic=mic, speaker=speaker)
    await the_call.connect()
    the_call.guard.coupling = 0.05            # badly under-reckoned: his own voice will stop him
    stop = threading.Event()
    noise = asyncio.create_task(room(the_call, speaker, mic, stop, coupling=1.0,
                                     caller=SPEECH, caller_at=4.0))
    try:
        the_call.line.mo_said(call.GREETING)
        await the_call.respond("Tell me something.")
        cut_short = the_call.cut_by_voice
        heard = await asyncio.wait_for(the_call.hear(), 25)
    finally:
        stop.set()
        noise.cancel()
        await asyncio.gather(noise, return_exceptions=True)
        await the_call.close()
    told = [m["content"] for m in the_call.line.transcript if m["role"] == "user"]
    return cut_short, heard, told, the_call.guard.coupling, out.getvalue()


def guard_cases():
    """The echo guard on its own: Mo's voice coming back, with and without a caller over it."""
    mo = np.frombuffer(SPEECH, np.int16).astype(np.float32)
    caller_voice = np.frombuffer(fixture("no.wav"), np.int16).astype(np.float32)
    rng = np.random.default_rng(3)
    step = speech.FRAME_SAMPLES

    def call_at(coupling, caller_at=None, gain=1.0, delay=0.06, gap=None):
        guard, ear = speech.EchoGuard(delay_ms=int(delay * 1000)), speech.Endpointer()
        t0 = 100.0
        playing = mo.copy()
        if gap:                                   # Mo pauses between sentences: nothing comes back
            playing[int(gap[0] * speech.MIC_RATE):int(gap[1] * speech.MIC_RATE)] = 0
        guard.plays(playing, speech.MIC_RATE, t0)
        mic = (rng.standard_normal(len(playing)) * 30
               + coupling * np.concatenate([np.zeros(int(delay * speech.MIC_RATE)), playing])[:len(playing)])
        if caller_at is not None:
            at = int(caller_at * speech.MIC_RATE)
            piece = caller_voice[:len(mic) - at] * gain
            mic[at:at + len(piece)] += piece
        for i in range(0, len(mic) - step + 1, step):
            frame = np.clip(mic[i:i + step], -32768, 32767).astype(np.int16).tobytes()
            if guard.hears_caller(frame, t0 + (i + step) / speech.MIC_RATE, ear):
                return round((i + step) / speech.MIC_RATE - (caller_at or 0), 2)
        guard.learn()
        return None

    return {
        "headphones, Mo only": call_at(0.03),
        "laptop speakers, Mo only": call_at(0.35),
        "everything comes back, Mo only": call_at(1.0),
        "headphones, caller talks over him": call_at(0.03, caller_at=1.0),
        "laptop speakers, caller talks over him": call_at(0.35, caller_at=1.0),
        "everything comes back, caller talks over him": call_at(1.0, caller_at=1.0),
        "caller in a gap between Mo's sentences": call_at(1.0, caller_at=1.2, gap=(1.1, 2.2)),
    }


def endpointer_cases():
    rng = np.random.default_rng(7)

    def fan(seconds, dbfs):          # broadband noise, the kind webrtcvad mistakes for speech
        x = np.cumsum(rng.standard_normal(int(16000 * seconds)))
        x -= np.convolve(x, np.ones(400) / 400, mode="same")
        return x / np.sqrt(np.mean(x ** 2)) * 32767 * 10 ** (dbfs / 20)

    def quiet(seconds):
        return rng.standard_normal(int(16000 * seconds)) * 32767 * 10 ** (-65 / 20)

    def pcm(x):
        return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

    def utterances(audio, prime=None):
        ep = speech.Endpointer()
        if prime is not None:
            ep.prime(speech.frames_of(pcm(prime)))
        return [u for u in (ep.feed(f) for f in speech.frames_of(pcm(audio))) if u]

    no = np.frombuffer(fixture("no.wav"), np.int16).astype(np.float32)
    loud = np.nonzero(np.abs(no) > 300)[0]
    no = no[loud[0]:loud[-1] + 1]                     # just the word: ~0.28 s
    caller_line = np.frombuffer(SPEECH, np.int16).astype(np.float32)
    click = np.zeros(1600)
    click[:960] = rng.standard_normal(960) * 12000
    fan_and_caller = fan(6.5, -30)
    fan_and_caller[16000:16000 + len(caller_line)] += caller_line
    return {
        "a lone 'No.' in a quiet room": len(utterances(np.concatenate([quiet(1), no, quiet(1.5)]))),
        "a click": len(utterances(np.concatenate([quiet(1), click, quiet(1.5)]))),
        "5 s of a loud fan": len(utterances(fan(5, -30), prime=fan(0.5, -30))),
        "a caller talking over a loud fan": len(utterances(fan_and_caller, prime=fan(0.5, -30))),
        "'No.', a 0.5 s pause, 'No.'": len(utterances(np.concatenate([quiet(1), no, quiet(0.5), no, quiet(1.5)]))),
        "'No.', a 1 s pause, 'No.'": len(utterances(np.concatenate([quiet(1), no, quiet(1.0), no, quiet(1.5)]))),
        "trailing silence sent on (s)": round(len(utterances(np.concatenate([quiet(1), no, quiet(2)]))[0]) / 32000
                                              - (0.3 + len(no) / 16000), 2),
    }


# ---------------------------------------------------------------- the checks
async def main():
    # ---- 1. a whole call: greeting, haggle, order, goodbye ----
    the_call, mic, screen = await scenario_full_call()
    assert the_call.ended_by == "mo", f"expected Mo to hang up after the caller's goodbye, got {the_call.ended_by}"
    assert the_call.ears.heard == ["Hi, it's Jane Doe.", "That's too much. Eighty dollars.", "Okay, deal.",
                                   "Thanks, bye!"], the_call.ears.heard
    first_words = [t for t in the_call.voice.spoken if t not in the_call.fixed_lines() or t == call.GREETING]
    assert first_words[0] == call.GREETING, first_words[:2]
    with db() as conn:
        orders = conn.execute("SELECT Amount, ListPrice, QuoteID FROM OrderDetails").fetchall()
    assert len(orders) == 1 and orders[0]["Amount"] == 108 and orders[0]["QuoteID"], [dict(o) for o in orders]
    print("1. full call: greeting -> Jane -> 80 dollars -> counter 108 -> deal -> order 108 -> Mo hangs up")

    # ---- 2. the line stays open while Mo talks, and the room is not mistaken for the caller ----
    assert mic.echo_offered > 0, "the test never put anything on the line while Mo talked"
    assert "cut in" not in screen, "the room's own hiss stopped Mo"
    assert len(the_call.ears.heard) == 4, the_call.ears.heard
    print(f"2. open line: {mic.echo_offered} frames of a quiet room went by while Mo talked - none of "
          "them stopped him, and none reached Whisper")

    # ---- 3. what the caller sees: their words, Mo's words, the order and the email ----
    for expected in ("You: Hi, it's Jane Doe.", "Mo:  Best I can do is 108 dollars.",
                     "(practice - not real): Cushioned trail runner for 108.00",
                     "The confirmation email (practice - not sent):", "Subject: Order 1 confirmed"):
        assert expected in screen, f"missing from the screen: {expected!r}\n{screen}"
    assert "🔧" not in screen and "📊" not in screen, "agent diagnostics leaked onto the call screen"
    print("3. screen: both sides of the conversation, then the order and the (unsent) confirmation email")

    # ---- 4. the line streams: the filler is heard while the model is still thinking ----
    pieces, total = await scenario_streaming()
    assert pieces and pieces[0][1] == voice_server.FILLER["customer_and_stock"], pieces
    assert pieces[0][0] < 0.5 <= 0.8 <= total, f"first words at {pieces[0][0]:.2f}s of {total:.2f}s: not streamed"
    print(f"4. streaming: \"{pieces[0][1]}\" reached the caller at {pieces[0][0]:.2f}s; the answer at {total:.2f}s")

    # ---- 5. Esc while listening hangs up at once ----
    esc_call, by = await scenario_hang_up_with_esc()
    assert by == "caller" and esc_call.state == "listening", (by, esc_call.state)
    print("5. Esc on the caller's turn: the call ends at once")

    # ---- 6. Enter while Mo talks cuts him off, and the call carries on ----
    hung, cut_screen, played, synthesised = await scenario_cut_in()
    assert "you cut in" in cut_screen, cut_screen
    assert not hung, "a cut-in must not end the call"
    # the greeting (whole), Mo's first sentence (whole), his second (cut off) - and not a sound after
    assert played == [True, True, False], f"clips played: {played}"
    assert "Then the mid boot." not in cut_screen, "sentences after the cut-in were still shown as spoken"
    # voice credits: at most one sentence synthesised beyond the one that was cut off
    assert len(synthesised) <= 3, f"synthesised {len(synthesised)} of 5 sentences for a reply cut off at 2"
    print(f"6. Enter while Mo talks: he stops mid-sentence 2 of 5; {len(synthesised)} of 5 synthesised; call goes on")

    # ---- 7. ...and the caller can talk straight away, while the rest of Mo's reply drains ----
    still_streaming, second, waited = await scenario_talk_after_cut_in()
    assert still_streaming, "the test meant to talk while Mo's reply was still streaming"
    assert second == "Actually, just the boots.", second
    print(f"7. after a cut-in the mic opens at once: words spoken while Mo's turn drained were heard "
          f"({waited:.2f}s to hand them over)")

    # ---- 7b. a pause between sentences does not end the caller's turn ----
    text, heard, first, second = await scenario_pause_mid_turn(0.9)
    merged = float(text.split(" seconds")[0])
    assert merged > first + second, f"the turn handed over {merged}s, but the caller said {first:.2f}s + {second:.2f}s"
    text, _, first_alone, _ = await scenario_pause_mid_turn(1.6)
    alone = float(text.split(" seconds")[0])
    assert alone < first_alone + 1.0, f"a 1.6 s silence should end the turn, but {alone}s went to Mo"
    for pause, whole in ((0.9, True), (1.6, False)):     # the same audio, arriving in one burst
        text, _, first_b, second_b = await scenario_pause_mid_turn(pause, burst=True)
        got = float(text.split(" seconds")[0])
        assert (got > first_b + second_b) == whole, f"burst, {pause}s pause: {got}s went to Mo"
    print(f"7b. a 0.9 s pause mid-turn: Mo got all of it as one turn ({merged}s); a 1.6 s silence ended the "
          f"turn ({alone}s) - and the same when the audio arrives in a burst")

    # ---- 8. a slow brain: Mo says "Bear with me a second." instead of dead air ----
    spoken, hold_screen = await scenario_hold_on()
    assert spoken[-2:] == [call.HOLD_ON, "The trail runner is 120 dollars."], spoken
    print("8. slow brain: \"Bear with me a second.\" fills the silence, then the answer")

    # ---- 9. a silent caller: nudged once, then Mo says goodbye and hangs up ----
    by, spoken, took = await scenario_silent_caller()
    assert by == "mo" and spoken[-2:] == [call.STILL_THERE, call.NO_ANSWER], (by, spoken[-3:])
    print(f"9. silent caller: \"{call.STILL_THERE}\" then \"{call.NO_ANSWER}\" and Mo hangs up ({took:.1f}s scaled)")

    # ---- 10. --keyboard: typing over Mo cuts him off and nothing typed is lost ----
    typed, transcript, kb_screen, kb_spoken = await scenario_keyboard()
    assert typed == "boots please", repr(typed)
    assert transcript[-2]["content"] == "boots please", transcript[-2:]
    assert "you cut in" in kb_screen and "Good choice." in kb_spoken, (kb_screen, kb_spoken)
    print("10. keyboard: typing over Mo cut him off; 'bo' + 'ots<BS>s please' reached Mo as 'boots please'")

    # ---- 11. the endpointer on hard audio ----
    cases = endpointer_cases()
    expected = {"a lone 'No.' in a quiet room": 1, "a click": 0, "5 s of a loud fan": 0,
                "a caller talking over a loud fan": 1, "'No.', a 0.5 s pause, 'No.'": 1,
                "'No.', a 1 s pause, 'No.'": 2}
    for case, want in expected.items():
        assert cases[case] == want, f"{case}: {cases[case]} utterances, expected {want}"
    assert cases["trailing silence sent on (s)"] <= 0.35, cases
    print("11. endpointer: " + "; ".join(f"{k} -> {v}" for k, v in cases.items()))

    # ---- 12. what the review found ----
    heard, vocabulary = await transcriber_keeps_short_answers()
    for short in ("Jane Doe.", "Eighty.", "Eighty dollars.", "One twenty.", "Cushioned trail runner."):
        assert heard[short] == short, f"{short!r} was thrown away as a prompt echo: {heard[short]!r}"
    assert heard[vocabulary] == "", "Whisper echoing its whole prompt got through as speech"
    print("12a. transcriber: 'Jane Doe.', 'Eighty.' and other short answers are kept; an echo of the prompt is not")

    unused, played, screen12 = await scenario_enter_while_thinking()
    assert not unused, "the test never pressed Enter while Mo was thinking"
    assert played == [True] and "The trail runner is 120 dollars." in screen12 and "cut in" not in screen12, \
        (played, screen12)
    print("12b. Enter while Mo is still thinking: ignored - his answer is heard in full")

    took, by = await scenario_esc_while_thinking()
    assert by == "caller" and took < 1.0, f"Esc took {took:.2f}s to hang up on a 3 s turn"
    print(f"12c. Esc while Mo is thinking (a 3 s turn): hung up in {took:.2f}s, not when his turn ended")

    text, spoken, screen12 = await scenario_still_here()
    assert text == "Tell me about your boots.", f"'Yes, I'm here.' reached the brain as the caller's answer: {text!r}"
    assert spoken[-2:] == [call.STILL_THERE, call.TAKE_YOUR_TIME] and "You: Yes, I'm here." in screen12, spoken[-3:]
    print(f"12d. 'Yes, I'm here.' after \"{call.STILL_THERE}\" is answered by the line (\"{call.TAKE_YOUR_TIME}\"), "
          "never passed to the brain as a yes")

    ok, played, screen12 = await scenario_voice_trouble(lasting=False)
    assert ok and played == [True, True] and "Third sentence." in screen12, (ok, played)
    ok, played, screen12 = await scenario_voice_trouble(lasting=True)
    assert not ok and played == [] and "Third sentence." in screen12 and "voice trouble" in screen12, (ok, played)
    print("12e. a passing voice failure costs one sentence's audio; a lasting one moves Mo to text for the call")

    assert await scenario_push_to_talk() == "I'll give you eighty dollars."
    print("12f. --push-to-talk: Enter, speak, Enter - heard as one turn")

    spoken, screen12 = await scenario_no_voice()
    assert spoken == [] and "Mo:  The trail runner is 120 dollars." in screen12, (spoken, screen12)
    print("12g. --no-voice: Mo's reply on screen, not one character of voice spent")

    hung, ok, screen12 = await scenario_speakers_die()
    assert not hung and not ok and "speakers stopped working" in screen12 and "Second sentence." in screen12, screen12
    print("12h. the speakers die mid-reply: Mo carries on in text instead of the call crashing")

    # ---- 12h2. --press-to-interrupt keeps the old half-duplex line: the mic is shut while Mo talks --
    _, agent = brain(ai("First sentence. Second sentence."))
    mic2, speaker2 = FakeMic(), CountingSpeaker()
    quiet_call, _ = new_call(agent, voice=FakeVoice(0.4), mic=mic2, speaker=speaker2,
                             press_to_interrupt=True)
    await quiet_call.connect()
    quiet_call.line.mo_said(call.GREETING)
    feeding = asyncio.create_task(caller(quiet_call, mic2, turns=0))
    reply = asyncio.create_task(quiet_call.respond("Tell me something."))
    while quiet_call.state != "speaking":
        await asyncio.sleep(0.01)
    for f in speech.frames_of(SPEECH):                 # his voice, straight back into the mic
        mic2.feed(f, echo=True)
    await reply
    feeding.cancel()
    await asyncio.gather(feeding, return_exceptions=True)
    assert quiet_call.guard is None and mic2.echo_dropped == mic2.echo_offered > 0, mic2.echo_offered
    assert speaker2.played == [True, True], speaker2.played
    await quiet_call.close()
    print("12h2. --press-to-interrupt: the mic is shut while Mo talks, so nothing of his comes back")

    # ---- 12i. talking over Mo stops him; his own voice coming back does not ----
    guards = guard_cases()
    for case, cut in guards.items():
        if "Mo only" in case:
            assert cut is None, f"{case}: Mo stopped himself after {cut}s"
        else:
            assert cut is not None and cut <= 0.45, f"{case}: stopped {cut}s after the caller spoke"
    print("12i. the echo guard: Mo never stops for his own voice (at 3%, 35% and 100% coming back), "
          f"and stops {max(v for v in guards.values() if v):.2f}s after a caller talks over him")

    hung, played, screen12, heard, coupling = await scenario_talk_over_mo(caller=None)
    assert played and all(played), f"Mo stopped himself in an echoing room: {played}"
    assert "cut in" not in screen12, screen12
    print(f"12j. a whole reply in a room that echoes him at 35%: all {len(played)} sentences played, "
          f"and he worked out the echo for himself ({coupling:.2f} of what plays comes back)")

    hung, played, screen12, heard, _ = await scenario_talk_over_mo(caller=SPEECH)
    assert False in played, f"talking over Mo did not stop him: {played}"
    assert "you cut in" in screen12 and not hung, screen12
    assert heard == "Actually, hold on.", heard
    print(f"12k. the caller talks over him: he stops mid-sentence ({played.count(True)} of "
          f"{len(REPLY)} finished) and what they said reaches Mo, first word and all")

    cut_short, heard, told, coupling, screen12 = await scenario_own_voice_back()
    assert cut_short, "the test meant to fool the guard into stopping Mo for his own voice"
    assert "Second sentence." not in told, f"Mo was told his own words: {told}"
    assert heard == "Actually, hold on." and coupling > 0.05, (heard, coupling)
    print(f"12l. when his own voice does come back as words, they are dropped, not answered - and he "
          f"expects more of himself back from then on (echo reckoned at {coupling:.2f})")

    # ---- 13. none of this touched the real shop ----
    from tools import wait_for_emails
    wait_for_emails()
    assert _safety.real_outbox_snapshot() == REAL_OUTBOX_BEFORE, "the test wrote into the real outbox"
    if REAL_DB_ORDERS is not None:
        now = sqlite3.connect(REAL_DB).execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0]
        assert now == REAL_DB_ORDERS, "the test placed an order in the real shoes.db"
    print("13. the real shoes.db and the real outbox are untouched")

    print("\nALL CALL ASSERTIONS PASSED")


asyncio.run(main())
