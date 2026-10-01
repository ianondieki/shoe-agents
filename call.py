"""Phone Mo's shoe shop from your terminal: talk into the microphone, hear Mo through the speakers.

    python call.py              a practice call: a copy of the shop, and no email ever leaves
    python call.py --check      test your microphone, speakers and the speech services first
    python call.py --keyboard   no microphone? type your side - Mo still talks back
    python call.py --live       the real shop: real orders, real confirmation emails

During the call: just talk, and pause when you're done. Talk over Mo and he stops to listen (Enter
does the same); Esc hangs up.

How it works: the terminal plays the part a phone network plays for the ElevenLabs agent. Your
speech becomes text (Groq Whisper), the text goes to Mo's brain - voice_server's own endpoint, run
inside this program, so prices, haggling, consent and orders behave exactly as on a real call -
and Mo's reply is spoken in his ElevenLabs voice a sentence at a time, as he comes up with it.
No tunnel, no dashboard, no agent-minutes.
"""
import argparse
import asyncio
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
PRACTICE = ROOT / ".practice"

# Mo's lines that come from this program rather than from his brain. All are cached on disk after
# the first call, so they play instantly and cost no voice credits again.
GREETING = "Thanks for calling Mo's shoe shop! Who am I speaking with?"
HOLD_ON = "Bear with me a second."
STILL_THERE = "Are you still there?"
TAKE_YOUR_TIME = "Good. Take your time."
NO_ANSWER = "I'll let you go. Call back any time. Bye!"
CLIENT_LINES = (GREETING, HOLD_ON, STILL_THERE, TAKE_YOUR_TIME, NO_ANSWER)
# After "Are you still there?", these answer the line, not Mo's last question.
_JUST_HERE = re.compile(r"^\W*(?:(?:yes|yeah|yep|yup|ya|sorry|hello|hi|i'?m (?:still )?here|i am (?:still )?here|"
                        r"still here|here|i'?m still on|still on the line|i'?m on the line)\W*)+$", re.I)

DIM, CYAN, YELLOW, GREEN, RED, BOLD = "90", "36", "33", "32", "31", "1"


# ======================================================================================
# 1. Mode. This runs BEFORE any project import: db.py reads DB_PATH the moment it loads.
# ======================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python call.py", description="Phone Mo's shoe shop from your terminal.")
    p.add_argument("--live", action="store_true",
                   help="use the real shop: orders are real and the customer is emailed")
    p.add_argument("--check", action="store_true",
                   help="test microphone, speakers and speech services, then exit")
    p.add_argument("--keyboard", action="store_true",
                   help="type your side of the call instead of speaking")
    p.add_argument("--push-to-talk", action="store_true",
                   help="press Enter to start talking and Enter again when you are done (for noisy rooms)")
    p.add_argument("--no-voice", action="store_true",
                   help="Mo replies in text only (saves ElevenLabs voice credits)")
    p.add_argument("--press-to-interrupt", action="store_true",
                   help="only Enter interrupts Mo (use it if he keeps stopping for his own voice)")
    p.add_argument("--patience", type=float, default=0.7, metavar="SECONDS",
                   help="how long a pause ends your turn (default 0.7; try 1.2 if Mo interrupts you)")
    p.add_argument("--input-device", help="microphone to use: a number or part of its name")
    p.add_argument("--output-device", help="speakers to use: a number or part of the name, or 'none'")
    return p.parse_args(argv)


def prepare_mode(live: bool) -> str:
    """Point the shop at the right database and email setting, before anything reads them."""
    if live:
        return "live"
    # Practice: today's real stock and prices, copied fresh for every call, so a practice order never
    # touches the shop. Emails are written to .practice/outbox and shown at the end - never sent.
    (PRACTICE / "outbox").mkdir(parents=True, exist_ok=True)
    practice_db = PRACTICE / "shoes.db"
    if (ROOT / "shoes.db").exists():
        shutil.copyfile(ROOT / "shoes.db", practice_db)
    elif practice_db.exists():
        practice_db.unlink()               # no shop yet: the brain seeds a fresh one here
    os.environ["DB_PATH"] = str(practice_db)
    os.environ["TRACE_PATH"] = str(PRACTICE / "traces.jsonl")
    os.environ["SMTP_USER"] = ""           # set before .env loads, so .env cannot turn email back on
    os.environ["SMTP_PASSWORD"] = ""
    os.environ["OUTBOX_DIR"] = str(PRACTICE / "outbox")
    return "practice"


def ensure_isolated():
    """Practice mode's promise, checked after the shop's modules have loaded, not assumed."""
    import db

    if (Path(db.DB_PATH).resolve() != (PRACTICE / "shoes.db").resolve()
            or Path(os.getenv("OUTBOX_DIR") or "outbox").resolve() != (PRACTICE / "outbox").resolve()
            or os.getenv("SMTP_USER") or os.getenv("SMTP_PASSWORD")):
        raise SystemExit("  Practice mode could not keep the call away from the real shop, "
                         "so it was not started.")


def preflight(args):
    """The keys a call cannot start without, said plainly - before anything rings."""
    if not args.keyboard and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("  GROQ_API_KEY is missing from .env, so Mo can't hear you. Get a free key at "
                         "console.groq.com/keys\n  and add it to .env - or type your side instead: "
                         "python call.py --keyboard")
    if not (os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY")):
        raise SystemExit("  Mo needs GROQ_API_KEY or MISTRAL_API_KEY in .env to think. See .env.example.")


# ======================================================================================
# 2. The screen and the keyboard. The brain's own chatter goes to a log, so the call reads cleanly.
# ======================================================================================
class Screen:
    """Both sides of the conversation, one line each, and a status line at the bottom."""

    def __init__(self, out=None):
        self.out = out or sys.__stdout__
        try:
            self.out.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        self.live = out is None and self.out.isatty()      # a real console: colours and a status line
        self.color = self.live and self._enable_ansi()
        self.status_shown = False

    @staticmethod
    def _enable_ansi() -> bool:
        if os.name != "nt":
            return True
        try:  # Windows 10's console needs virtual-terminal processing switched on for colours
            import ctypes

            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            return bool(k32.GetConsoleMode(h, ctypes.byref(mode))
                        and k32.SetConsoleMode(h, mode.value | 0x0004))
        except Exception:
            return False

    def paint(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def _clear(self):
        if self.status_shown:
            self.out.write("\r\x1b[K" if self.color else "\r" + " " * 79 + "\r")
            self.status_shown = False

    def line(self, text: str = ""):
        self._clear()
        self.out.write(text + "\n")
        self.out.flush()

    def status(self, text: str):
        if not self.live:
            return
        self._clear()
        if text:
            self.out.write("\r" + text)
            self.status_shown = True
        self.out.flush()

    def you(self, text):
        self.line(self.paint(CYAN, "  You: ") + text)

    def mo(self, text):
        self.line(self.paint(YELLOW, "  Mo:  ") + text)

    def note(self, text):
        self.line(self.paint(DIM, f"       ({text})"))

    # typing, for --keyboard
    def prompt(self, typed: str):
        self._clear()
        self.out.write(self.paint(CYAN, "  You: ") + typed)
        self.out.flush()

    def echo(self, text: str):
        self.out.write(text)
        self.out.flush()

    def banner(self, mode: str, keyboard: bool):
        self.line()
        if mode == "practice":
            self.line(self.paint(BOLD, "  Mo's shoe shop") + self.paint(GREEN, "  - practice call"))
            self.line(self.paint(DIM, "  A copy of the shop: nothing you order is real, and no email is sent."))
        else:
            self.line(self.paint(BOLD, "  Mo's shoe shop") + self.paint(RED, "  - LIVE call"))
            self.line(self.paint(RED, "  The real shop: orders are real and the customer is emailed."))
        if keyboard:
            self.line(self.paint(DIM, "  Type your side and press Enter. Typing while Mo talks cuts him off. "
                                      "Esc hangs up."))
        else:
            self.line(self.paint(DIM, "  Talk normally and pause when you're done. Talk over Mo any time "
                                      "and he stops. Esc hangs up."))
        self.line()


def key_pressed() -> str | None:
    """A key pressed without Enter, on Windows; None elsewhere or when nothing was pressed."""
    if os.name != "nt":
        return None
    import msvcrt

    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two characters
            msvcrt.getwch()
            return None
        return ch
    return None


def flush_keys():
    """Forget keys pressed before now: an Enter typed while the phone rang must not cut off the greeting."""
    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()


# ======================================================================================
# 3. Audio devices: the microphone and the speakers.
# ======================================================================================
def _drain(q: queue.Queue) -> list:
    """Everything waiting in a queue, right now."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def find_device(spec, kind: str):
    """A device by number or by part of its name; None means the Windows default."""
    import sounddevice as sd

    if spec is None:
        return None
    if str(spec).isdigit():
        return int(spec)
    matches = [i for i, d in enumerate(sd.query_devices())
               if spec.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0]
    if not matches:
        raise SystemExit(f"  No {kind} device matches {spec!r}. See the list with: python call.py --check")
    # prefer the MME variant: it accepts 16 kHz / 24 kHz and resamples for us
    mme = [i for i in matches if sd.query_hostapis(sd.query_devices(i)["hostapi"])["name"] == "MME"]
    return (mme or matches)[0]


class Mic:
    """16 kHz mono 30 ms frames into a queue - but only while unmuted. It starts muted and is opened
    only on the caller's turn: laptop speakers leak into the microphone, and Mo must never hear
    (and answer) himself."""

    def __init__(self, device=None):
        import numpy as np
        import sounddevice as sd
        from speech import FRAME_SAMPLES, MIC_RATE

        self.np = np
        self.q: queue.Queue = queue.Queue(maxsize=3000)
        self.level = 0.0
        self.muted = threading.Event()
        self.muted.set()
        self.decimate = 1
        rate = MIC_RATE
        try:
            sd.check_input_settings(device=device, samplerate=MIC_RATE, channels=1, dtype="int16")
        except Exception:
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
            if rate % MIC_RATE:
                raise
            self.decimate = rate // MIC_RATE
        self.stream = sd.RawInputStream(samplerate=rate, blocksize=FRAME_SAMPLES * self.decimate,
                                        dtype="int16", channels=1, device=device, callback=self._frame)
        self.stream.start()

    def _frame(self, indata, frames, when, status):
        a = self.np.frombuffer(indata, dtype=self.np.int16)
        if self.decimate > 1:
            a = a.reshape(-1, self.decimate).mean(axis=1).astype(self.np.int16)
        self.level = float(self.np.sqrt(self.np.mean(a.astype(self.np.float32) ** 2)))
        if not self.muted.is_set():
            try:
                # the capture time, not the time it is read: the echo guard lines frames up with
                # what the speaker was playing, and a busy loop must not shift that
                self.q.put_nowait((time.monotonic(), a.tobytes()))
            except queue.Full:
                pass

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def sample(self, seconds: float) -> list[bytes]:
        """Blocking: record a moment of the room (to learn how quiet it is), then mute again."""
        self.drain()
        self.muted.clear()
        time.sleep(seconds)
        self.muted.set()
        frames = []
        while not self.q.empty():
            frames.append(self.q.get_nowait()[1])
        return frames

    def close(self):
        self.stream.stop()
        self.stream.close()


class SpeakerBroken(Exception):
    """The speakers stopped working mid-call (a headset unplugged, say): Mo carries on in text."""


class Speaker:
    """Plays Mo's 24 kHz audio in 100 ms slices, so cutting in takes effect within a tenth of a second."""

    def __init__(self, device=None, silent=False):
        from speech import VOICE_RATE

        self.rate, self.silent, self.latency = VOICE_RATE, silent, 0.0
        self.guard = None                  # the echo guard, once there is a microphone to protect
        self.halt = threading.Event()      # set by close(): whatever is playing stops at the next slice
        self.lock = threading.Lock()       # held while a slice is written: close() never pulls the stream mid-write
        if not silent:
            import sounddevice as sd

            self.stream = sd.RawOutputStream(samplerate=VOICE_RATE, channels=1, dtype="int16", device=device)
            self.stream.start()
            self.latency = float(self.stream.latency)   # audio still in the device after write() returns

    def play(self, audio, stop: threading.Event) -> bool:
        """Blocking. True if it played to the end, False if it was cut off."""
        if self.guard is not None and len(audio):
            self.guard.plays(audio, self.rate, time.monotonic() + self.latency)
        step = self.rate // 10
        for i in range(0, len(audio), step):
            with self.lock:
                if self.halt.is_set():
                    return False
                try:
                    if stop.is_set():
                        if not self.silent:
                            self.stream.abort()   # drop what is already queued in the device
                            self.stream.start()
                        if self.guard is not None:    # the rest never played: expect none of it back
                            self.guard.silence(time.monotonic() + self.latency)
                        return False
                    piece = audio[i:i + step]
                    if self.silent:
                        time.sleep(len(piece) / self.rate)
                    else:
                        self.stream.write(piece.tobytes())
                except Exception as e:
                    raise SpeakerBroken(f"{type(e).__name__}: {e}") from e
        return True

    def close(self):
        self.halt.set()
        with self.lock:                    # waits for the slice being written: a tenth of a second at most
            if not self.silent:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass


def ring_until(speaker: Speaker, connected: threading.Event):
    """A phone ringing (two short rings, then a pause) until the line connects. Blocking."""
    import numpy as np

    r = speaker.rate
    t = np.arange(int(r * 0.4)) / r
    burst = 0.12 * 32767 / 2 * (np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 450 * t))
    burst *= np.minimum(1, np.minimum(t, t[::-1]) / 0.01)          # 10 ms fades: no clicks
    ring = np.concatenate([burst, np.zeros(int(r * 0.2)), burst]).astype(np.int16)
    pause = np.zeros(int(r * 2.0), dtype=np.int16)
    while True:
        speaker.play(ring, threading.Event())           # a ring is never cut in half
        if connected.is_set() or not speaker.play(pause, connected):
            return


# ======================================================================================
# 4. The line to Mo: voice_server's endpoint, run inside this process.
# ======================================================================================
class StreamingASGITransport(httpx.AsyncBaseTransport):
    """Calls an ASGI app in-process and hands over each piece of the body the moment it is sent.

    httpx's own ASGITransport collects the whole body first (it yields b"".join(body) once the app
    has returned), which would turn Mo's sentence-at-a-time reply into one lump at the end of the
    turn - and deliver "Let me have a look." together with the answer it exists to cover.
    """

    def __init__(self, app):
        self.app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        url = request.url
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": request.method, "scheme": url.scheme, "path": url.path,
            "raw_path": url.raw_path.split(b"?")[0], "query_string": url.query, "root_path": "",
            "headers": [(k.lower(), v) for k, v in request.headers.raw],
            "server": (url.host, url.port or 80), "client": ("127.0.0.1", 0),
        }
        chunks: asyncio.Queue = asyncio.Queue()
        started, finished = asyncio.Event(), asyncio.Event()
        head: dict = {}
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()     # Starlette listens for a disconnect while it streams: never early
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                head["status"], head["headers"] = message["status"], message.get("headers", [])
                started.set()
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    chunks.put_nowait(bytes(message["body"]))
                if not message.get("more_body", False) and not finished.is_set():
                    finished.set()
                    chunks.put_nowait(None)

        def ended(task: asyncio.Task):
            # the app crashed or was cancelled: end the body rather than leave the reader waiting
            if not task.cancelled():
                task.exception()          # retrieved here, so asyncio does not log it a second time
            if not finished.is_set():
                finished.set()
                chunks.put_nowait(None)
            started.set()

        task = asyncio.create_task(self.app(scope, receive, send))
        task.add_done_callback(ended)
        await started.wait()
        if "status" not in head:
            await task                    # re-raises whatever stopped the app from answering
            raise RuntimeError("the app finished without answering")

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                while (chunk := await chunks.get()) is not None:
                    yield chunk

            async def aclose(self):
                if not task.done():
                    task.cancel()

        return httpx.Response(head["status"], headers=head["headers"], stream=Body(), request=request)


class Line:
    """Speaks to Mo exactly as ElevenLabs does - the whole transcript every turn, SSE streamed back -
    but in-process: no port, no tunnel, no network between the caller and the brain."""

    def __init__(self, agent):
        import voice_server

        self.app = voice_server.create_app(agent=agent)
        self.client = httpx.AsyncClient(transport=StreamingASGITransport(self.app),
                                        base_url="http://mo.local", timeout=120)
        self.session = f"terminal-{uuid.uuid4().hex[:12]}"
        self.transcript: list[dict] = []
        self.headers = {"x-session-id": self.session}
        secret = os.getenv("VOICE_SHARED_SECRET", "")
        if secret:
            self.headers["authorization"] = f"Bearer {secret}"

    def mo_said(self, text: str):
        self.transcript.append({"role": "assistant", "content": text})

    async def turn(self, text: str):
        """Yields ("say", sentence) as Mo produces them, then ("hangup", None) if he ends the call."""
        self.transcript.append({"role": "user", "content": text})
        spoken, hangup = [], False
        body = {"model": "mo", "stream": True, "messages": list(self.transcript)}
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=body,
                                          headers=self.headers) as r:
                r.raise_for_status()
                async for raw in r.aiter_lines():
                    if not raw.startswith("data: ") or raw == "data: [DONE]":
                        continue
                    delta = json.loads(raw[6:])["choices"][0]["delta"]
                    if delta.get("content"):
                        spoken.append(delta["content"])
                        yield "say", delta["content"]
                    if any(tc["function"]["name"] == "end_call" for tc in delta.get("tool_calls") or []):
                        hangup = True
        finally:
            self.transcript.append({"role": "assistant", "content": "".join(spoken).strip()})
        if hangup:
            yield "hangup", None

    async def aclose(self):
        await self.client.aclose()


def _build_agent():
    import voice_server

    return voice_server.build_server_agent()


# ======================================================================================
# 5. The call.
# ======================================================================================
class HangUp(Exception):
    """The call is over: the caller hung up (Esc, Ctrl+C), or Mo gave up on a silent line."""

    def __init__(self, by: str = "caller"):
        super().__init__(by)
        self.by = by


class Call:
    """One phone call. Every part can be handed in (tests do); whatever is not is built in connect()."""

    HOLD_ON_AFTER_S = 6       # Mo silent this long after the caller spoke: "Bear with me a second."
    NUDGE_AFTER_S = 15        # caller silent this long: "Are you still there?"
    GIVE_UP_AFTER_S = 45      # ...and this long in all: Mo says goodbye and hangs up
    HOLD_TURN_S = 0.5         # silence past the end of an utterance before it goes to Mo (see hear)

    def __init__(self, args, mode: str, screen: Screen, *, line=None, ears=None, voice=None,
                 mic=None, speaker=None):
        self.args, self.mode, self.screen = args, mode, screen
        self.state = "connecting"
        self.started = time.time()
        self.voice_ok = not args.no_voice
        self.line, self.ears, self.voice, self.mic, self.speaker = line, ears, voice, mic, speaker
        self.ear = None               # the endpointer: kept for the whole call, it knows the room
        self.ended_by = None          # "mo" or "caller", once the call is over
        self.typed: list[str] = []    # --keyboard: keys typed while Mo was talking
        self.mic_open = False         # opened early, when the caller cut Mo off
        self.opening = None           # ...by this task, which hear() waits for rather than opening twice
        self.guard = None             # tells the caller's voice from Mo's, coming back into the mic
        self.carry: list[bytes] = []  # the caller's first words, spoken over Mo, on their way to Whisper
        self.cut_by_voice = False     # this turn started by talking over him, not by pressing Enter
        self.last_aside = ""          # Mo's last line of his own ("are you still there?"), for the echo check
        self.voice_misses = 0         # passing voice failures in a row
        self.credits_before = self.credits_after = None
        self.summarised = False

    # ---------- setup ----------
    async def connect(self):
        import speech

        s = self.screen
        s.status(s.paint(DIM, "  Calling Mo's shoe shop..."))
        if self.speaker is None:
            out = self.args.output_device
            self.speaker = Speaker(None if out in (None, "none") else find_device(out, "output"),
                                   silent=(out == "none"))
        if self.mic is None and not self.args.keyboard:
            try:
                self.mic = Mic(find_device(self.args.input_device, "input"))
            except SystemExit:
                raise
            except Exception as e:
                raise SystemExit(f"  Could not open a microphone ({type(e).__name__}: {e}).\n"
                                 "  Try: python call.py --check    or type instead: python call.py --keyboard")
        if self.mic is not None and self.ear is None:
            self.ear = speech.Endpointer(end_silence_ms=int(self.args.patience * 1000))
            if hasattr(self.mic, "sample"):
                # Half a second of the room before anything plays: what "quiet" means here.
                self.ear.prime(await asyncio.to_thread(self.mic.sample, 0.5))
        if self.mic is not None and not self.args.press_to_interrupt:
            # The line stays open while Mo talks, so the caller can cut in by talking. The guard is
            # what makes that safe: it knows what the speaker is playing and how much of it comes back.
            self.guard = speech.EchoGuard()
            self.speaker.guard = self.guard

        connected = threading.Event()
        ringing = learning = None
        if self.voice_ok and not self.speaker.silent:
            ringing = asyncio.create_task(asyncio.to_thread(ring_until, self.speaker, connected))
            if self.guard is not None:
                # The ringing is sound from the same speakers: how much of it comes back is a first
                # measure of the room, so the caller can talk over Mo from his very first sentence.
                self.mic.drain()
                self.mic.muted.clear()
                learning = asyncio.create_task(self._learn_from_ring(connected))
        try:
            await self._connect_parts()
        finally:
            connected.set()
            for t in (ringing, learning):
                if t is not None:
                    await t
            self._mute()
        s.status("")

    async def _learn_from_ring(self, connected: threading.Event):
        """Listen to the phone ringing through the speakers: that is Mo's own sound coming back."""
        while not connected.is_set():
            try:
                when, frame = self.mic.q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            self.guard.hears_caller(frame, when, self.ear)     # for the ratios, not to stop anything
        self.guard.learn()
        self.guard.first_words()                               # nothing here belongs to the caller
        print(f"[{time.time():.2f}] the room sends back {self.guard.coupling:.2f} of what plays")

    async def _connect_parts(self):
        import speech

        if self.line is None:
            self.line = Line(await asyncio.to_thread(_build_agent))    # a few seconds, once
        import voice_server

        if self.ears is None and not self.args.keyboard:
            self.ears = speech.Transcriber(vocabulary=await asyncio.to_thread(speech.shop_vocabulary))
        if self.voice is None:
            self.voice = speech.Voice()
        await speech.warm(self.ears, self.voice)
        if not self.voice_ok:
            return
        self.credits_before = await self.voice.credits_left()
        # Mo's fixed lines, synthesised once and cached for good: instant, and free from then on.
        # Two at a time: the free plan makes only a few voices at once.
        fixed = [*CLIENT_LINES, voice_server.FALLBACK_REPLY, *voice_server.FILLER.values()]
        two = asyncio.Semaphore(2)

        async def cache(text):
            async with two:
                await self.voice.speak(text, cache=True)

        results = await asyncio.gather(*(cache(t) for t in fixed), return_exceptions=True)
        for r in results:
            if isinstance(r, speech.SpeechUnavailable):
                if r.lasting:
                    self.voice_ok = False
                    self.screen.note(str(r))
                    return
                # a passing problem: that line is simply made when it is first needed
            elif isinstance(r, BaseException):
                raise r

    def fixed_lines(self) -> set[str]:
        import voice_server

        return {*CLIENT_LINES, voice_server.FALLBACK_REPLY, *voice_server.FILLER.values()}

    # ---------- speaking ----------
    def _voice_trouble(self, e):
        """A lasting voice problem (no credits, no permission) moves Mo to text for the rest of the call;
        a passing one (a timeout, a busy server) costs only the sentence it hit."""
        self.voice_misses += 1
        if e.lasting or self.voice_misses >= 3:
            self.voice_ok = False
            self.screen.note(str(e) if e.lasting else "Mo's voice keeps dropping out - he'll reply in text")
        elif self.voice_misses == 1:
            self.screen.note(str(e))

    async def _play(self, audio, stop: threading.Event) -> bool:
        """Play one clip in a worker thread. False if it was cut off."""
        try:
            return await asyncio.to_thread(self.speaker.play, audio, stop)
        except SpeakerBroken as e:
            print(f"The speakers failed: {e}")               # into call.log
            self.voice_ok = False
            self.screen.note("the speakers stopped working - Mo will reply in text")
            return True

    async def speak_one(self, text: str, stop: threading.Event | None = None) -> bool:
        """Say one of Mo's fixed lines (text on screen as it is spoken). False if cut off."""
        import speech

        self.screen.mo(text)
        if not self.voice_ok:
            return True
        try:
            audio = await self.voice.speak(text, cache=True)
            self.voice_misses = 0
        except speech.SpeechUnavailable as e:
            self._voice_trouble(e)
            return True
        return await self._play(audio, stop or threading.Event())

    async def respond(self, text: str) -> bool:
        """One turn of Mo's: his reply streams in a sentence at a time and each sentence is spoken as
        soon as it is ready, while the next is being synthesised. True when Mo has ended the call."""
        import speech
        import voice_server

        fixed = self.fixed_lines()
        stop = threading.Event()
        hangup = cut = caller_hung_up = False
        replied = asyncio.Event()
        sentences: asyncio.Queue = asyncio.Queue()
        audio: asyncio.Queue = asyncio.Queue()
        # The sentence playing and the next one ready - never more: a caller who cuts in must not
        # have spent voice credits on sentences nobody will hear.
        ahead = asyncio.Semaphore(2)
        self._open_or_mute()
        self.state = "thinking"
        self.screen.status(self.screen.paint(DIM, "  Mo is thinking..."))

        async def produce():
            nonlocal hangup
            try:
                async for kind, payload in self.line.turn(text):
                    if kind == "say" and payload.strip():
                        replied.set()
                        await sentences.put(payload.strip())
                    elif kind == "hangup":
                        hangup = True
            except Exception:
                print("The line to Mo failed:\n" + traceback.format_exc())      # into call.log
                replied.set()
                await sentences.put(voice_server.FALLBACK_REPLY)
            finally:
                replied.set()
                await sentences.put(None)

        async def bear_with_me():
            # A rate-limited free model can take a while; silence on a phone line feels like a drop.
            try:
                await asyncio.wait_for(replied.wait(), self.HOLD_ON_AFTER_S)
            except asyncio.TimeoutError:
                await sentences.put(HOLD_ON)

        async def synthesise():
            while (s := await sentences.get()) is not None:
                await ahead.acquire()
                clip = None
                if self.voice_ok and not stop.is_set():
                    try:
                        clip = await self.voice.speak(s, cache=s in fixed)
                        self.voice_misses = 0
                    except speech.SpeechUnavailable as e:
                        self._voice_trouble(e)
                await audio.put((s, clip))
            await audio.put(None)

        async def play():
            nonlocal cut
            while (item := await audio.get()) is not None:
                s, clip = item
                try:
                    if stop.is_set():
                        continue
                    self.state = "speaking"
                    self.screen.mo(s)
                    if clip is not None and self.voice_ok:
                        if not self.args.keyboard:
                            self.screen.status(self.screen.paint(
                                DIM, "  (talk over him any time)" if self.guard is not None
                                else "  (Enter cuts Mo off)"))
                        if not await self._play(clip, stop):
                            cut = True
                finally:
                    ahead.release()

        def cut_in(by_voice: bool):
            """Mo stops talking, mid-word, and the line is the caller's."""
            if stop.is_set():
                return
            stop.set()
            self.screen.note("you cut in" if self.args.keyboard else "you cut in - go ahead")
            if self.mic is None:
                return
            if by_voice:
                # The mic is already open and their first words are in it: keep them for Whisper.
                self.carry, self.cut_by_voice, self.mic_open = self.guard.first_words(), True, True
            else:
                self.opening = asyncio.create_task(self._open_mic_early())

        async def listen_for_keys():
            nonlocal caller_hung_up
            while True:
                k = key_pressed()
                if k is None:
                    await asyncio.sleep(0.03)
                    continue
                if k in ("\x1b", "\x03"):
                    # Hang up now, not when Mo's turn is over: stop reading his reply. The server
                    # finishes the turn on its own, so an order that was already going through still does.
                    caller_hung_up = True
                    stop.set()
                    work.cancel()
                    return
                if self.args.keyboard:
                    if k == "\x08":
                        if self.typed:
                            self.typed.pop()
                        continue
                    if k.isprintable():
                        self.typed.append(k)     # the start of their next line: kept, not lost
                # Only a Mo who is talking can be cut off. An Enter while he is still thinking (a habit
                # after speaking) would silently throw away an answer the caller never heard.
                if self.state != "speaking" or stop.is_set():
                    continue
                if k in ("\r", "\n", " ") or (self.args.keyboard and k.isprintable()):
                    cut_in(by_voice=False)

        # Mo's reply is read to the end even when the caller cut in: the server finishes the turn either
        # way (an order half-placed must still commit), and one turn at a time keeps the server's view
        # of the call in order.
        work = asyncio.gather(produce(), synthesise(), play())
        keys = asyncio.create_task(listen_for_keys())
        hold = asyncio.create_task(bear_with_me())
        ears = (asyncio.create_task(self._watch_for_voice(stop, lambda: cut_in(by_voice=True)))
                if self.guard is not None else None)
        try:
            await work
        except asyncio.CancelledError:
            if not caller_hung_up:
                raise
        finally:
            stop.set()                     # an audio thread still mid-sentence stops within 0.1 s
            watching = [t for t in (keys, hold, ears) if t is not None]
            for t in watching:
                t.cancel()
            await asyncio.gather(*watching, return_exceptions=True)
            if self.guard is not None:
                # only a turn nobody talked over says anything about the echo
                self.guard.learn(clean=not self.cut_by_voice)
        self.screen.status("")
        if caller_hung_up:
            raise HangUp("caller")
        return hangup and not cut

    def _mute(self):
        if self.mic is not None:
            self.mic.muted.set()
        self.mic_open = False

    def _open_or_mute(self):
        """While Mo talks: with a guard the line stays open, so the caller can cut in by talking;
        without one (no microphone, or --press-to-interrupt) it is muted so he never hears himself."""
        self.carry = []
        if self.guard is None or self.mic is None:
            self._mute()
            return
        self.mic.drain()              # Mo's tail from the last turn; from here the guard reads the line
        self.mic.muted.clear()
        self.mic_open = False         # open for the guard, not yet for the caller's turn

    async def _watch_for_voice(self, stop: threading.Event, on_cut) -> None:
        """While Mo talks the line stays open: talk over him and he stops. Frames that carry only his
        own voice back are dropped; about 150 ms of the caller's voice is enough."""
        while not stop.is_set():
            try:
                when, frame = self.mic.q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if self.state == "speaking" and self.guard.hears_caller(frame, when, self.ear):
                on_cut()
                return

    def _mo_just_said(self, text: str) -> bool:
        """Were those Mo's own words, coming back? Only asked of a turn that stopped him by voice: if
        most of what was heard sits word for word inside what he was saying, the room said it, not the
        caller. (A caller who really does repeat him says more than just his words.)"""
        import difflib

        said = " ".join(m["content"] for m in self.line.transcript[-2:] if m["role"] == "assistant")
        heard = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        hay = re.sub(r"[^a-z0-9 ]", " ", (said + " " + self.last_aside).lower())
        heard, hay = " ".join(heard.split()), " ".join(hay.split())
        if len(heard) < 6 or not hay:
            return False
        same = difflib.SequenceMatcher(None, hay, heard).find_longest_match(0, len(hay), 0, len(heard))
        return same.size >= 0.8 * len(heard)

    def _take_carry(self):
        """The caller's first words, spoken over Mo, reach the endpointer before anything new."""
        if not self.carry:
            return
        self.ear.reset()
        for frame in self.carry:
            self.ear.feed(frame)
        self.carry = []

    async def _open_mic_early(self):
        """After a cut-in: open the mic as soon as Mo's voice has stopped, not when his turn is over,
        so the caller's first words are not lost while the rest of the reply drains."""
        await asyncio.sleep(0.12 + self.speaker.latency)
        self.mic.drain()
        self.ear.reset()
        self.mic.muted.clear()
        self.mic_open = True
        print(f"[{time.time():.2f}] listening (after a cut-in)")      # call.log: when the mic opened

    async def _say_aside(self, text: str):
        """One of Mo's own lines while the caller is quiet ("Are you still there?"). The brain never
        sees these: they are the line's, not the conversation's. The caller can talk over these too."""
        stop = threading.Event()
        self._open_or_mute()
        self.state = "speaking"
        self.last_aside = text

        def answered():
            stop.set()
            self.carry, self.cut_by_voice, self.mic_open = self.guard.first_words(), True, True

        watching = (asyncio.create_task(self._watch_for_voice(stop, answered))
                    if self.guard is not None else None)
        try:
            await self.speak_one(text, stop)
        finally:
            if watching is not None:
                watching.cancel()
                await asyncio.gather(watching, return_exceptions=True)
                self.guard.learn(clean=not self.cut_by_voice)
        if self.mic_open:                  # they answered over him: their words are already in hand
            self.mic_open = False
            self.state = "listening"
            self._take_carry()
        else:
            await self._reopen_mic()

    async def _reopen_mic(self):
        await asyncio.sleep(0.25 + self.speaker.latency)     # the tail of Mo's last word, still in the air
        self.mic.drain()
        self.ear.reset()
        self.mic.muted.clear()
        self.state = "listening"
        print(f"[{time.time():.2f}] listening")               # call.log: when the mic opened

    # ---------- listening ----------
    async def hear(self) -> str:
        """Wait for the caller to say something; return it as text."""
        if self.args.keyboard:
            return await self.type_line()
        import speech

        if self.opening is not None:        # a cut-in is already opening the mic: let it, don't do it twice
            await asyncio.gather(self.opening, return_exceptions=True)
            self.opening = None
        if self.mic_open:
            self.mic_open = False
            self.state = "listening"
            self._take_carry()              # they talked over Mo: their first words are already in hand
            print(f"[{time.time():.2f}] listening (they cut in)")
        else:
            await self._reopen_mic()
        quiet_since, nudged, last_meter = time.monotonic(), False, 0.0
        push_to_talk = self.args.push_to_talk
        recording, taken = not push_to_talk, []
        # The end of an utterance is not always the end of the caller's turn: people pause between
        # sentences for 0.8 s and more ("That's a bit steep. ... I'll give you eighty."). So the mic
        # stays open while Whisper works on what was said; if they carry on, that transcription is
        # dropped and everything is heard together. Nothing reaches Mo until the silence has lasted
        # this long past the utterance's end - time Whisper needs anyway, so it costs almost nothing.
        # The hold is counted in audio - frames of silence heard - not in time spent: a program that falls
        # behind and catches up in a burst must still tell a 1.6 s silence from a 0.9 s pause. (A mic that
        # stops sending frames altogether is covered by the clock.)
        said, pending, ended_at, was_speaking, quiet_frames = b"", None, 0.0, False, 0
        hold_frames = int(self.HOLD_TURN_S * 1000 / speech.FRAME_MS)
        while True:
            if pending is not None and (quiet_frames >= hold_frames
                                        or time.monotonic() - ended_at >= self.HOLD_TURN_S + 1.5):
                # The turn is over; only the words may still be on their way from Whisper.
                text, pending = await pending, None
                if text and self.cut_by_voice and self.guard is not None and self._mo_just_said(text):
                    # He stopped for his own voice and it came back as words. Never pass those on -
                    # Mo would answer himself - and expect more of him back from here on.
                    print(f"[{time.time():.2f}] that was Mo's own voice: {text!r}; echo now reckoned "
                          f"at {self.guard.coupling:.2f} of what plays")
                    self.guard.false_cut()
                    self.cut_by_voice, text = False, ""
                if text and nudged and _JUST_HERE.match(text):
                    # "Yes, I'm here" answers "are you still there?" - a line the brain never heard.
                    # Passed on, it would read as a yes to Mo's last real question ("shall I put it
                    # through?"), so the line answers it itself.
                    self.screen.you(text)
                    await self._say_aside(TAKE_YOUR_TIME)
                    text, nudged = "", False
                if text:
                    self._mute()
                    self.cut_by_voice = False
                    self.state = "thinking"         # the turn is Mo's now
                    return text
                if self.cut_by_voice and self.guard is not None:
                    # Mo was stopped and there were no words in it: that was his own voice coming
                    # back. Expect more of it next time, so he stops talking for the caller only.
                    self.guard.false_cut()
                    print(f"[{time.time():.2f}] cut in for nothing: echo now reckoned at "
                          f"{self.guard.coupling:.2f} of what plays")
                    self.cut_by_voice = False
                said = b""                          # nothing usable in it: the turn starts afresh
                quiet_since = time.monotonic()
            k = key_pressed()
            if k in ("\x1b", "\x03"):
                if pending is not None:
                    pending.cancel()
                raise HangUp("caller")
            if push_to_talk and k in ("\r", "\n"):
                if not recording:
                    recording, taken = True, []
                    self.mic.drain()
                else:
                    while True:                     # what the mic heard up to the Enter belongs to this turn
                        try:
                            taken.append(self.mic.q.get_nowait()[1])
                        except queue.Empty:
                            break
                    pcm = b"".join(taken)
                    recording = False
                    if len(pcm) < speech.MIC_RATE * 2 * 0.3:
                        self.screen.status("  Too short - press Enter, speak, then press Enter again.")
                        continue
                    self._mute()
                    self.screen.status(self.screen.paint(DIM, "  ..."))
                    text = await self._transcribe(pcm)
                    if text:
                        self.state = "thinking"
                        return text
                    await self._reopen_mic()
                continue
            try:
                _, frame = self.mic.q.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                if now - last_meter > 0.1:
                    self._meter(recording, working=pending is not None)
                    last_meter = now
                if not push_to_talk and not self.ear.speaking and not said:
                    if not nudged and now - quiet_since > self.NUDGE_AFTER_S:
                        # for call.log: when "Mo can't hear me", these numbers say why
                        print(f"[{time.time():.2f}] nudge: no speech for {now - quiet_since:.0f}s; "
                              f"room floor {self.ear.floor:.0f}, "
                              f"loudest of the last 3s {max(self.ear.levels, default=0):.0f}, "
                              f"mic level {self.mic.level:.0f}")
                        nudged = True
                        await self._say_aside(STILL_THERE)
                        quiet_since = time.monotonic()
                    elif nudged and now - quiet_since > self.GIVE_UP_AFTER_S - self.NUDGE_AFTER_S:
                        await self._say_aside(NO_ANSWER)
                        raise HangUp("mo")
                await asyncio.sleep(0.01)
                continue
            if push_to_talk:
                if recording:
                    taken.append(frame)
                continue
            utterance = self.ear.feed(frame)
            if self.ear.speaking:
                quiet_since = time.monotonic()
                if pending is not None:             # they carried on: that was a pause, not the end
                    pending.cancel()
                    pending = None
            elif was_speaking and not utterance and said and pending is None:
                # a cough or a click after the pause: what they had said still stands
                pending, ended_at, quiet_frames = asyncio.create_task(self._transcribe(said)), time.monotonic(), 0
            elif pending is not None:
                quiet_frames += 1
            was_speaking = self.ear.speaking
            if utterance:
                print(f"[{time.time():.2f}] utterance: {len(utterance) / 32000:.2f}s; "
                      f"room floor {self.ear.floor:.0f}")
                said += utterance
                pending, ended_at, quiet_frames = asyncio.create_task(self._transcribe(said)), time.monotonic(), 0

    async def type_line(self) -> str:
        """--keyboard: the caller's line, typed. Whatever they typed while Mo talked is already there."""
        s = self.screen
        self.state = "listening"
        if os.name != "nt":             # no msvcrt: a plain input() on a daemon thread
            typed: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: typed.put(sys.__stdin__.readline()), daemon=True).start()
            s.prompt("")
            while typed.empty():
                await asyncio.sleep(0.05)
            text = typed.get().strip()
        else:
            buf, self.typed = self.typed, []
            s.prompt("".join(buf))
            while True:
                k = key_pressed()
                if k is None:
                    await asyncio.sleep(0.02)
                    continue
                if k in ("\x1b", "\x03"):
                    s.echo("\n")
                    raise HangUp("caller")
                if k in ("\r", "\n"):
                    if "".join(buf).strip():
                        s.echo("\n")
                        break
                elif k == "\x08":
                    if buf:
                        buf.pop()
                        s.echo("\b \b")
                elif k.isprintable():
                    buf.append(k)
                    s.echo(k)
            text = "".join(buf).strip()
        if text.lower() in ("quit", "exit", "/quit", "/exit"):
            raise HangUp("caller")
        return text

    async def _transcribe(self, pcm: bytes) -> str:
        """The words in pcm, or "" (with a note to the caller) when there are none to be had."""
        import speech

        try:
            text = await self.ears.text(pcm)
        except speech.SpeechUnavailable as e:
            self.screen.note(str(e))
            return ""
        except Exception as e:
            print("Transcription failed:\n" + traceback.format_exc())
            self.screen.note(f"couldn't make that out ({type(e).__name__}) - please say it again")
            return ""
        if not text:
            self.screen.note("didn't catch that - go ahead")
        return text

    def _meter(self, recording: bool, working: bool = False):
        lvl = min(int(self.mic.level / 180), 12)
        bar = "#" * lvl + "-" * (12 - lvl)
        if self.args.push_to_talk:
            msg = (f"  [{bar}]  recording - press Enter when you're done" if recording
                   else "  Press Enter and speak, then Enter again when you're done.")
        elif self.ear.speaking:
            msg = f"  [{bar}]  hearing you..."
        elif working:
            msg = f"  [{bar}]  ..."
        else:
            msg = f"  [{bar}]  your turn"
        self.screen.status(self.screen.paint(CYAN if self.ear.speaking else DIM, msg))

    # ---------- the call itself ----------
    async def run(self):
        s = self.screen
        s.banner(self.mode, self.args.keyboard)
        await self.connect()
        if self.credits_before:
            used, limit = self.credits_before
            s.line(s.paint(DIM, f"  (Mo's voice: {limit - used:,} of {limit:,} ElevenLabs credits left this month)"))
            s.line()
        flush_keys()
        self._mute()
        self.state = "speaking"
        await self.speak_one(GREETING)
        self.line.mo_said(GREETING)
        while True:
            said = await self.hear()
            if not self.args.keyboard:
                s.you(said)
            if await self.respond(said):
                self.ended_by = "mo"
                s.note("Mo hung up")
                return

    async def close(self):
        if self.opening is not None and not self.opening.done():
            self.opening.cancel()
        if self.voice_ok and self.voice is not None and getattr(self.voice, "chars_used", 0):
            try:
                self.credits_after = await asyncio.wait_for(self.voice.credits_left(), 5)
            except Exception:
                pass
        for c in (self.mic, self.speaker):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
        for c in (self.line, self.ears, self.voice):
            if c is not None:
                try:
                    await c.aclose()
                except Exception:
                    pass

    # ---------- afterwards ----------
    def summary(self):
        if self.summarised:
            return
        self.summarised = True
        s = self.screen
        secs = int(time.time() - self.started)
        who = {"mo": "Mo hung up", "caller": "you hung up"}.get(self.ended_by, "call ended")
        s.line()
        s.line(s.paint(BOLD, f"  Call over ({who}) after {secs // 60}m {secs % 60:02d}s."))
        if not self.line:
            return
        import tools
        from db import db

        tools.wait_for_emails(timeout=20)
        with db() as conn:
            orders = conn.execute(
                """SELECT o.OrderID, o.Amount, o.ListPrice, s.StyleDesc, c.CustomerName, c.Email
                   FROM PriceQuote q JOIN OrderDetails o ON o.OrderID = q.OrderID
                   JOIN ShoeInventory s ON s.ShoeID = o.ShoeID JOIN CustomerInfo c ON c.CustomerID = o.CustomerID
                   WHERE q.SessionID LIKE ? AND q.Status = 'USED'""",
                (f"voice:{self.line.session}%",)).fetchall()
        if not orders:
            s.line("  No order was placed.")
        for o in orders:
            saved = f", {o['ListPrice'] - o['Amount']:.2f} off the list price" if o["ListPrice"] > o["Amount"] else ""
            real = " (practice - not real)" if self.mode == "practice" else ""
            s.line(s.paint(GREEN, f"  Order {o['OrderID']}{real}: {o['StyleDesc']} for {o['Amount']:.2f}{saved}"
                                  f" - {o['CustomerName']}"))
            ok, what = tools.EMAIL_RESULTS.get(o["OrderID"], (None, ""))
            if ok and what.startswith("DRY RUN") and "saved to " in what:
                mail = Path(what.split("saved to ", 1)[1].strip())
                heading = ("  The confirmation email (practice - not sent):" if self.mode == "practice"
                           else "  Email isn't set up in .env, so the confirmation was saved, not sent:")
                s.line(s.paint(DIM, heading))
                if mail.exists():
                    for text_line in mail.read_text(encoding="utf-8").splitlines():
                        s.line(s.paint(DIM, f"     {text_line}"))
            elif ok:
                s.line(s.paint(DIM, f"  Confirmation emailed to {o['Email']}."))
            elif ok is False:
                s.line(s.paint(RED, f"  The confirmation email did not go out: {what}"))
            else:
                s.line(s.paint(DIM, "  The confirmation email is still on its way."))
        if self.credits_before and self.credits_after:
            used = self.credits_after[0] - self.credits_before[0]
            left = self.credits_after[1] - self.credits_after[0]
            s.line(s.paint(DIM, f"  Mo's voice used {used:,} ElevenLabs credits ({left:,} left this month)."))
        s.line()


# ======================================================================================
# 6. --check: find out before the call whether everything works.
# ======================================================================================
async def check(args, screen: Screen) -> bool:
    import sounddevice as sd

    import speech

    s = screen
    ok = True
    good, bad, meh = s.paint(GREEN, "  ok "), s.paint(RED, "  !! "), s.paint(YELLOW, "  !  ")
    s.line()
    s.line(s.paint(BOLD, "  Checking your setup for a call with Mo"))
    s.line()
    for key, what in (("GROQ_API_KEY", "Mo's brain, and hearing you"),
                      ("ELEVENLABS_API_KEY", "Mo's voice (without it, he replies in text)")):
        present = bool(os.getenv(key))
        ok &= present or key == "ELEVENLABS_API_KEY"
        s.line(f"{good if present else bad} {key:<20} {what}" + ("" if present else "  - add it to .env"))

    din, dout = sd.default.device
    s.line()
    s.line(s.paint(DIM, "  Microphones:"))
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] and sd.query_hostapis(d["hostapi"])["name"] == "MME":
            s.line(f"    [{i}] {d['name']}" + (s.paint(GREEN, "   <- default") if i == din else ""))
    s.line(s.paint(DIM, "  Speakers:"))
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] and sd.query_hostapis(d["hostapi"])["name"] == "MME":
            s.line(f"    [{i}] {d['name']}" + (s.paint(GREEN, "   <- default") if i == dout else ""))
    s.line(s.paint(DIM, "  (use others with --input-device N and --output-device N)"))

    voice = speech.Voice()
    s.line()
    if os.getenv("ELEVENLABS_API_KEY"):
        used = await voice.credits_left()
        if used:
            s.line(f"{good} ElevenLabs: {used[1] - used[0]:,} of {used[1]:,} voice credits left this month")
        try:
            clip = await voice.speak("If you can hear me, your speakers are working.", cache=True)
            if args.output_device != "none":
                spk = Speaker(find_device(args.output_device, "output"))
                s.line("  Playing Mo's voice - you should hear him now...")
                # ...and listening to him at the same time: how much of Mo comes back into the
                # microphone decides whether the caller can cut in by simply talking over him.
                listener = None
                if not args.keyboard:
                    try:
                        listener = Mic(find_device(args.input_device, "input"))
                        listener.muted.clear()
                    except Exception:
                        listener = None
                await asyncio.to_thread(spk.play, clip, threading.Event())
                spk.close()
                if listener is not None:
                    back = [speech.rms(f) for _, f in _drain(listener.q)]
                    listener.close()
                    played = [speech.rms(clip[i:i + 720].tobytes()) for i in range(0, len(clip) - 720, 720)]
                    loud = sorted(v for v in played if v > 100)
                    heard = sorted(v for v in back if v > 0)
                    if loud and heard:
                        echo = heard[len(heard) // 2] / loud[len(loud) // 2]
                        if echo < 0.25:
                            s.line(f"{good} Mo hardly reaches your microphone ({echo:.0%} of him comes "
                                   "back): talk over him any time and he will stop")
                        elif echo < 0.8:
                            s.line(f"{good} Mo comes back into your microphone at {echo:.0%}: talking "
                                   "over him should stop him (headphones make it certain)")
                        else:
                            s.line(f"{meh} Mo comes back into your microphone almost as loudly as you "
                                   f"({echo:.0%}). Use headphones, or press Enter to interrupt him.")
        except speech.SpeechUnavailable as e:
            s.line(f"{bad} {e}")
        except Exception as e:
            ok = False
            s.line(f"{bad} Could not play sound: {e}")
    await voice.aclose()

    if args.keyboard:
        s.line()
        s.line(s.paint(GREEN, "  Ready - start a call with: python call.py --keyboard"))
        s.line()
        return ok

    s.line()
    s.line("  Now say something for 4 seconds - for example: \"Hi, it's Jane Doe\"")
    try:
        mic = Mic(find_device(args.input_device, "input"))
    except SystemExit:
        raise
    except Exception as e:
        s.line(f"{bad} Could not open the microphone: {e}")
        s.line("     Plug one in, or type instead: python call.py --keyboard")
        return False
    mic.muted.clear()
    peak, frames, t0 = 0.0, [], time.time()
    while time.time() - t0 < 4:
        try:
            frames.append(mic.q.get(timeout=0.1)[1])
        except queue.Empty:
            pass
        peak = max(peak, mic.level)
        s.status(f"     [{'#' * min(int(mic.level / 180), 12):<12}]  {4 - int(time.time() - t0)}s")
    mic.close()
    s.status("")
    if peak == 0:
        ok = False
        s.line(f"{bad} The microphone gave pure silence - Windows is probably blocking it:")
        s.line("     Settings > Privacy > Microphone > turn on 'Allow desktop apps to access your microphone'")
    elif peak < 150:
        s.line(f"{meh} Very quiet. Speak closer, or raise the level in Windows sound settings.")
    ear = speech.Endpointer()
    heard_pcm = next((u for u in (ear.feed(f) for f in frames) if u), None)
    if heard_pcm is None and ear.speaking:        # still talking when the 4 seconds ran out
        heard_pcm = b"".join(ear.frames)
    if heard_pcm and os.getenv("GROQ_API_KEY"):
        ears = speech.Transcriber(vocabulary=speech.shop_vocabulary())
        try:
            heard = await ears.text(heard_pcm)
        except speech.SpeechUnavailable as e:
            ok, heard = False, ""
            s.line(f"{bad} {e}")
        await ears.aclose()
        if heard:
            s.line(f"{good} Mo heard you say: \"{heard}\"")
        else:
            s.line(f"{meh} Heard sound, but no words. Try again a little closer to the mic.")
    elif peak:
        s.line(f"{meh} No speech in those 4 seconds. Run the check again and talk during it.")
    s.line()
    s.line(s.paint(GREEN, "  Ready - start a call with: python call.py") if ok
           else s.paint(RED, "  Fix the items above, then run: python call.py --check"))
    s.line()
    return ok


# ======================================================================================
# 7. Entry point.
# ======================================================================================
def confirm_live(screen: Screen) -> bool:
    screen.line()
    screen.line(screen.paint(RED, "  LIVE call: the real shop. An order is real, and the customer is emailed."))
    screen.echo("  Start it? [y/N] ")
    try:
        return sys.__stdin__.readline().strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)            # "shoes.db", "outbox" and ".env" mean this project's, wherever you ran it from
    mode = prepare_mode(args.live)
    screen = Screen()

    # The brain's own chatter - tool calls, costs, warnings - belongs in a log, not across the call.
    log_dir = PRACTICE if mode == "practice" else ROOT
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "call.log"
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {mode} call =====\n")
    sys.stdout = sys.stderr = log

    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        if args.check:
            return 0 if asyncio.run(check(args, screen)) else 1
        if mode == "live" and not confirm_live(screen):
            screen.line("  Not started.")
            return 0
        preflight(args)
        if mode == "practice":
            ensure_isolated()
        call = Call(args, mode, screen)

        async def go():
            try:
                await call.run()
            except HangUp as h:
                call.ended_by = h.by
                if h.by == "caller":
                    screen.note("you hung up")
            except asyncio.CancelledError:           # Ctrl+C
                asyncio.current_task().uncancel()
                call.ended_by = "caller"
                screen.note("you hung up")
            finally:
                await call.close()
            call.summary()
            # Whatever the brain was still doing when the call ended - a model call the caller hung up
            # on - is not worth waiting for: any order and its email are settled (summary waited).
            import tools

            if not any(t.is_alive() for t in tools._EMAIL_THREADS):
                sys.__stdout__.flush()
                log.flush()
                os._exit(0)

        try:
            asyncio.run(go())
        except KeyboardInterrupt:
            call.ended_by = call.ended_by or "caller"
            screen.note("you hung up")
        call.summary()
        return 0
    except SystemExit as e:
        if e.code not in (None, 0, 1):
            screen.line()
            screen.line(str(e.code))
            screen.line()
        return 1
    except Exception:
        print(traceback.format_exc())
        screen.line()
        screen.line(screen.paint(RED, f"  Something went wrong - the details are in {log_path}"))
        screen.line()
        return 1


if __name__ == "__main__":
    sys.exit(main())
