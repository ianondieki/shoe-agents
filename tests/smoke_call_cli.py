"""LIVE, black-box: run `python call.py` exactly as a person would, and be the caller.

The program runs as its own process, in practice mode, started from outside the project folder;
its microphone and speakers are the two ends of VB-Audio Virtual Cable (see tests/_caller.py). The
caller answers when the program opens its microphone (it logs each opening), as a person answers
when Mo stops talking.
Nothing plays on your speakers, and nothing real is ordered or emailed - practice mode itself
guarantees that, and this test checks it did. Spends ~100-300 ElevenLabs credits and Groq quota.

    python tests/smoke_call_cli.py
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import difflib
import os
import sqlite3
import subprocess
import sys
import threading
import time

import sounddevice as sd

from _caller import LINES, caller_voice, cable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAN = ["hello", "price", "offer", "deal", "bye"]


def main():
    voices = caller_voice()
    speak_into, listen_on = cable()
    real_orders = sqlite3.connect(os.path.join(ROOT, "shoes.db")).execute(
        "SELECT COUNT(*) FROM OrderDetails").fetchone()[0]
    real_outbox = _safety.real_outbox_snapshot()

    # When is it the caller's turn? The program writes "[time] listening" to its log each time it opens
    # the microphone - the one moment a person would hear Mo stop and know to answer. (Guessing it from
    # the sound on the line raced: the screen shows Mo's words before they are heard.)
    log_path = os.path.join(ROOT, ".practice", "call.log")
    start = os.path.getsize(log_path) if os.path.exists(log_path) else 0

    def mic_opened() -> int:
        try:
            with open(log_path, "rb") as f:
                f.seek(start)
                return f.read().decode("utf-8", errors="replace").count("] listening")
        except FileNotFoundError:
            return 0

    mouth = sd.RawOutputStream(samplerate=16000, channels=1, dtype="int16", device=speak_into)
    mouth.start()

    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "call.py"),
         "--input-device", str(listen_on), "--output-device", str(speak_into)],
        cwd=os.path.expanduser("~"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    screen: list[str] = []

    def read():
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            screen.append(line)

    threading.Thread(target=read, daemon=True).start()

    def wait(cond, timeout, what):
        t0 = time.monotonic()
        while not cond():
            if time.monotonic() - t0 > timeout or proc.poll() is not None:
                raise SystemExit(f"FAILED waiting for {what}")
            time.sleep(0.05)

    def said(prefix, since=0):
        return sum(line.lstrip().startswith(prefix) for line in screen[since:])

    t0, opened = time.monotonic(), 0
    try:
        for key in PLAN:
            wait(lambda: mic_opened() > opened, 120, "Mo to finish and listen")
            opened = mic_opened()
            time.sleep(0.6)                              # a person takes a moment to answer
            heard = said("You:")
            mouth.write(voices[key])
            time.sleep(float(mouth.latency))
            wait(lambda: said("You:") > heard, 30, f"Mo to hear {key!r}")
        proc.wait(timeout=90)
    finally:
        if proc.poll() is None:
            proc.kill()
        mouth.stop()
        mouth.close()
    time.sleep(0.3)

    text = "\n".join(screen)
    print("\n================ CHECKS ================")
    print(f"python call.py ran {time.monotonic() - t0:.0f}s and exited with {proc.returncode}")
    assert proc.returncode == 0, proc.returncode
    assert "practice call" in text and "(Mo hung up)" in text, "no practice banner, or Mo did not hang up"
    assert said("You:") == len(PLAN), f"{said('You:')} caller turns for {len(PLAN)} lines"
    heard = [line.split("You:", 1)[1].strip() for line in screen if line.lstrip().startswith("You:")]
    for got, key in zip(heard, PLAN):              # each line heard whole: no clipped start, no half turn
        close = difflib.SequenceMatcher(None, got.lower(), LINES[key].lower()).ratio()
        assert close >= 0.8, f"heard {got!r} for {LINES[key]!r} ({close:.2f})"
    assert "(practice - not real)" in text, "no practice order in the summary"
    assert "The confirmation email (practice - not sent):" in text, "the unsent email was not shown"
    assert "Traceback" not in text and "🔧" not in text and "📊" not in text, "internals on the call screen"
    print("ok  a practice call from the real command: heard every line, haggled, ordered, Mo hung up")

    practice = sqlite3.connect(os.path.join(ROOT, ".practice", "shoes.db"))
    last = practice.execute("SELECT OrderID, Amount, ListPrice FROM OrderDetails ORDER BY OrderID DESC").fetchone()
    assert last and last[1] < last[2], last
    print(f"ok  order {last[0]} at {last[1]:.2f} (list {last[2]:.2f}) is in .practice/shoes.db, the copy")

    assert sqlite3.connect(os.path.join(ROOT, "shoes.db")).execute(
        "SELECT COUNT(*) FROM OrderDetails").fetchone()[0] == real_orders, "the REAL shop got an order"
    assert _safety.real_outbox_snapshot() == real_outbox, "the REAL outbox changed"
    print("ok  the real shoes.db and the real outbox are untouched")
    print("\nCLI SMOKE CALL PASSED")


if __name__ == "__main__":
    main()
