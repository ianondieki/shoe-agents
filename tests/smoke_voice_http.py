"""LIVE: a real haggle over real HTTP against the running voice server and the real models.

Start the server first, against a throwaway database AND with email switched off - this script
only drives the server, so the server's own environment decides whether mail goes out:
    set DB_PATH=tests\\_tmp\\live.db && set TRACE_PATH=tests\\_tmp\\live_trace.jsonl && set SMTP_USER= && set OUTBOX_DIR=tests\\_tmp\\outbox && python voice_server.py
then:
    python tests/smoke_voice_http.py

Plays the platform's part exactly: sends the whole transcript every turn, streams the reply,
and measures time-to-first-word from the caller's side of the socket. Spends free-tier quota.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import json
import os
import statistics
import sys
import time

import httpx

URL = os.getenv("VOICE_URL", "http://127.0.0.1:8013")
SECRET = os.getenv("VOICE_SHARED_SECRET", "")
SESSION = f"smoke-{int(time.time())}"

CALLER = [
    "Hi, it's Jane Doe here, I'm after some running shoes.",
    "How much is the cushioned trail runner?",
    "That's a bit steep. I'll give you eighty for it.",
    "Alright, I'll take it at the price you just said.",
    "Great, email me the confirmation please.",
    "That's everything, thanks. Bye!",
]


def wait_for_server(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{URL}/health", timeout=2).json().get("ok"):
                return
        except Exception:
            pass
        time.sleep(0.5)
    sys.exit(f"server at {URL} not ready after {timeout}s")


def turn(client, transcript):
    """Send the transcript, stream the reply. Returns (text, first_word_ms, total_ms, hung_up)."""
    headers = {"authorization": f"Bearer {SECRET}"} if SECRET else {}
    body = {"model": "shoe", "stream": True, "messages": transcript,
            "elevenlabs_extra_body": {"conversation_id": SESSION}}
    t0 = time.perf_counter()
    first, text, hung_up = None, [], False
    with client.stream("POST", f"{URL}/v1/chat/completions", json=body, headers=headers) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if delta.get("content"):
                if first is None:
                    first = int((time.perf_counter() - t0) * 1000)
                text.append(delta["content"])
            if any(tc["function"]["name"] == "end_call" for tc in delta.get("tool_calls") or []):
                hung_up = True
    return "".join(text), first, int((time.perf_counter() - t0) * 1000), hung_up


wait_for_server()
transcript = [{"role": "assistant", "content": "Thanks for calling Mo's, who am I speaking with?"}]
firsts, totals = [], []
with httpx.Client(timeout=60) as client:
    for i, said in enumerate(CALLER, 1):
        transcript.append({"role": "user", "content": said})
        reply, first, total, hung_up = turn(client, transcript)
        transcript.append({"role": "assistant", "content": reply})
        firsts.append(first or total)
        totals.append(total)
        print(f"\n[{i}] Caller: {said}\n    Mo:     {reply}\n    first word {first}ms · done {total}ms"
              + ("  · HUNG UP" if hung_up else ""))
        time.sleep(5)  # a caller's natural pause, and headroom under the rate limit

print("\n================ OVER THE WIRE ================")
print(f"first word ms : median={statistics.median(firsts):.0f}  max={max(firsts)}  all={firsts}")
print(f"full reply ms : median={statistics.median(totals):.0f}  max={max(totals)}")
print(f"hung up at end: {hung_up}")
