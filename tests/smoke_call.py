"""LIVE: a whole terminal phone call with real audio, and nobody at the keyboard.

Needs VB-Audio Virtual Cable (a free virtual audio device: https://vb-audio.com/Cable/). The caller
is a Windows voice (offline, free) played into "CABLE Input"; Mo listens on "CABLE Output" as his
microphone and speaks into "CABLE Input" too - so every word Mo says goes straight back into his own
microphone: the worst echo there is. Nothing plays on your speakers.

Real: PortAudio devices, the endpointer, Groq Whisper, Mo's brain on the real models, ElevenLabs
voice. Spends a little free-tier quota (Groq, and ~300-600 ElevenLabs credits).
Safe: a copy of the shop, and email forced off (tests/_safety.py) - no order or email is real.

    python tests/smoke_call.py
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import asyncio
import difflib
import os
import shutil
import sqlite3
import statistics
import sys
import threading
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
DB = os.path.join(SCRATCH, "smoke_call.db")
REAL_DB = os.path.join(ROOT, "shoes.db")
shutil.copyfile(REAL_DB, DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "smoke_call_trace.jsonl")
os.environ["VOICE_SHARED_SECRET"] = ""
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))       # API keys; the blanked SMTP settings stay blank
assert os.environ["SMTP_USER"] == "" and os.environ["SMTP_PASSWORD"] == "", "email must be off"

from _caller import LINES, cable, caller_voice  # noqa: E402

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

import call  # noqa: E402
import speech  # noqa: E402
import tools  # noqa: E402

REAL_OUTBOX_BEFORE = _safety.real_outbox_snapshot()
REAL_ORDERS_BEFORE = sqlite3.connect(REAL_DB).execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0]


class TimedSpeaker(call.Speaker):
    """Mo's real speaker (into the cable), noting when each clip started and whether it finished."""
    def __init__(self, device):
        super().__init__(device)
        self.starts: list[float] = []
        self.finished: list[bool] = []

    def play(self, audio, stop):
        self.starts.append(time.perf_counter())
        done = super().play(audio, stop)
        self.finished.append(done)
        return done


class Ears(speech.Transcriber):
    """The real transcriber, keeping every utterance it was handed (tests/_tmp/heard_N.wav) and what
    it made of it - including ones dropped because the caller carried on talking."""
    heard: list[tuple[str, float, str]] = []

    async def text(self, pcm):
        n = len(self.heard) + 1
        with wave.open(os.path.join(SCRATCH, f"heard_{n}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        self.heard.append((f"heard_{n}.wav", round(len(pcm) / 32000, 2), "(dropped: the caller went on)"))
        said = await super().text(pcm)
        self.heard[n - 1] = (f"heard_{n}.wav", round(len(pcm) / 32000, 2), said)
        return said


def orders_now() -> int:
    return sqlite3.connect(DB).execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0]


async def main():
    voices = caller_voice()
    cable_in, cable_out = cable()
    print(f"devices: caller + Mo speak into [{cable_in}] {sd.query_devices(cable_in)['name']}; "
          f"Mo listens on [{cable_out}] {sd.query_devices(cable_out)['name']}")

    args = call.parse_args(["--input-device", str(cable_out)])
    speaker = TimedSpeaker(cable_in)
    screen = call.Screen(out=sys.__stdout__)
    the_call = call.Call(args, "practice", screen, speaker=speaker)
    the_call.ears = Ears(vocabulary=speech.shop_vocabulary())
    orders_before = orders_now()

    mouth = sd.RawOutputStream(samplerate=16000, channels=1, dtype="int16", device=cable_in)
    mouth.start()
    lags, said_lines, cut = [], [], []

    def speak(key):
        mouth.write(voices[key])
        time.sleep(float(mouth.latency))

    async def caller():
        plan = ["hello", "barge", "price", "offer", "deal"]
        extra_yes = 2
        while True:
            if plan[:1] == ["barge"]:
                # Talk over Mo, mid-sentence, the way a person cuts in - he must stop and listen.
                plan.pop(0)
                while the_call.state != "speaking":
                    if the_call.ended_by:
                        return
                    await asyncio.sleep(0.02)
                await asyncio.sleep(1.5)                   # let him get going first
                cut[:] = [len(speaker.finished)]
                said_lines.append(LINES["hold"])
                told = sum(m["role"] == "user" for m in the_call.line.transcript)
                await asyncio.to_thread(speak, "hold")
                while sum(m["role"] == "user" for m in the_call.line.transcript) == told:
                    await asyncio.sleep(0.02)              # he has heard it...
                while the_call.state != "listening":
                    await asyncio.sleep(0.02)              # ...and answered it
                continue
            while the_call.state != "listening":
                if the_call.ended_by or the_call.state == "over":
                    return
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.6)                       # a person takes a moment to answer
            if plan:
                key = plan.pop(0)
            elif orders_now() == orders_before and extra_yes:
                extra_yes -= 1
                key = "yes"
            else:
                key = "bye"
            said_lines.append(LINES[key])
            await asyncio.to_thread(speak, key)
            done_talking = time.perf_counter()
            n = len(speaker.starts)
            while the_call.state == "listening":
                await asyncio.sleep(0.02)
            while len(speaker.starts) == n and not the_call.ended_by:
                await asyncio.sleep(0.02)
            if len(speaker.starts) > n:
                lags.append(speaker.starts[n] - done_talking)
            if key == "bye":
                return

    async def run_call():
        try:
            await the_call.run()
        finally:
            the_call.state = "over"

    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.gather(run_call(), caller()), timeout=300)
    finally:
        await the_call.close()
        mouth.stop()
        mouth.close()
    the_call.summary()
    took = time.perf_counter() - t0

    # ---------------- what must be true ----------------
    tools.wait_for_emails()
    sys.stdout = sys.__stdout__
    print("\n================ CHECKS ================")
    print(f"call length {took:.0f}s")
    for f, secs, said in Ears.heard:
        print(f"   whisper got {f} ({secs}s): {said!r}")
    if lags:
        print(f"caller stops talking -> Mo's voice starts: median {statistics.median(lags):.2f}s, "
              f"max {max(lags):.2f}s, all {[round(x, 2) for x in lags]}")
    assert the_call.ended_by == "mo", f"Mo should hang up after the caller's goodbye (ended by {the_call.ended_by})"
    print("ok  Mo hung up after the caller said goodbye")

    # What Mo was told the caller said: each caller line, whole, once - and never Mo's own voice
    # coming back through his microphone.
    told = [m["content"] for m in the_call.line.transcript if m["role"] == "user"]
    assert len(told) == len(said_lines), f"caller said {len(said_lines)} lines, Mo got {len(told)} turns: {told}"
    for got, meant in zip(told, said_lines):
        close = difflib.SequenceMatcher(None, got.lower().replace("$80", "eighty dollars"), meant.lower()).ratio()
        enough = 0.5 if meant == LINES["hold"] else 0.8      # that one was said over Mo's own voice
        assert close >= enough, f"Mo was told {got!r} for {meant!r} ({close:.2f})"
    print(f"ok  all {len(said_lines)} caller lines reached Mo whole, and nothing else - no echo, no half sentences")

    # ...including the one said over him: he stopped mid-sentence and listened
    assert cut and False in speaker.finished[cut[0]:], "talking over Mo did not stop him mid-sentence"
    over_him = told[said_lines.index(LINES["hold"])]
    print(f"ok  the caller talked over Mo: he stopped mid-sentence and heard {over_him!r}")

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    assert orders_now() == orders_before + 1, f"expected one new order, got {orders_now() - orders_before}"
    o = dict(con.execute("SELECT * FROM OrderDetails ORDER BY OrderID DESC LIMIT 1").fetchone())
    assert o["Amount"] < o["ListPrice"], o
    print(f"ok  order {o['OrderID']}: {o['Amount']:.2f} (list {o['ListPrice']:.2f}) - haggled down, in the COPY of the shop")

    ok, what = tools.EMAIL_RESULTS.get(o["OrderID"], (None, ""))
    assert ok and what.startswith("DRY RUN") and os.path.abspath(SCRATCH) in os.path.abspath(what.split("saved to ", 1)[1]), what
    print(f"ok  confirmation dry-run to {what.split('saved to ', 1)[1]} - not sent")

    assert sqlite3.connect(REAL_DB).execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0] == REAL_ORDERS_BEFORE
    assert _safety.real_outbox_snapshot() == REAL_OUTBOX_BEFORE
    print("ok  the real shoes.db and the real outbox are untouched")
    print("\nSMOKE CALL PASSED")


if __name__ == "__main__":
    log = open(os.path.join(SCRATCH, "smoke_call.log"), "w", encoding="utf-8", buffering=1)
    real_stdout = sys.stdout
    sys.stdout = log                    # the brain's own chatter; the call itself prints to the console
    try:
        asyncio.run(main())
    finally:
        sys.stdout = real_stdout
