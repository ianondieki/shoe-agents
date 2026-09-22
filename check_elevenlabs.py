"""Show what the ElevenLabs key in .env is allowed to do, one permission scope at a time.

A restricted key answers 401 "missing_permissions" naming the exact scope it lacks, so probing
one endpoint per scope maps the key precisely. Every probe is free: reads, or writes with a
deliberately invalid body that fails validation before anything is created or synthesised.

    python check_elevenlabs.py
"""
import os

try:
    # See agent_core.py: antivirus/proxy TLS interception breaks certifi-based verification.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE = "https://api.elevenlabs.io"
RACHEL = "21m00Tcm4TlvDq8ikWAM"  # a stock voice id, so the TTS probe reaches the permission check

# (scope, why this project wants it, method, path, request kwargs)
PROBES = [
    ("convai_read", "NEEDED  list agents, read call transcripts and latency", "GET", "/v1/convai/agents", {}),
    ("convai_write", "NEEDED  create/configure the agent, store the Custom LLM secret", "POST", "/v1/convai/agents/create", {"json": {}}),
    ("user_read", "NEEDED  see credits and agent minutes left (15 min/month is tight)", "GET", "/v1/user/subscription", {}),
    ("voices_read", "NEEDED  list voices to choose one for Mo", "GET", "/v1/voices", {}),
    ("models_read", "useful  confirm eleven_flash_v2_5 is available", "GET", "/v1/models", {}),
    ("text_to_speech", "useful  pre-synthesise filler clips; the self-hosted pipeline later", "POST", f"/v1/text-to-speech/{RACHEL}", {"json": {"text": ""}}),
    ("speech_to_text", "optional  the platform does its own; only for a self-hosted pipeline", "POST", "/v1/speech-to-text", {"data": {"model_id": "scribe_v1"}}),
    ("speech_history_read", "not needed", "GET", "/v1/history?page_size=1", {}),
]


def main():
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set in .env")
    headers = {"xi-api-key": key}
    granted, subscription, agents, voices = [], None, None, None

    print(f"{'scope':<22}{'status':<10}why this project wants it")
    print("-" * 100)
    for scope, why, method, path, kw in PROBES:
        try:
            r = httpx.request(method, BASE + path, headers=headers, timeout=20, **kw)
        except Exception as e:
            print(f"{scope:<22}{'NETWORK':<10}{type(e).__name__}: {str(e)[:60]}")
            continue
        # Success bodies vary by endpoint (/v1/models is a bare list); only errors carry "detail".
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        detail = body.get("detail") if isinstance(body, dict) else None
        status = detail.get("status") if isinstance(detail, dict) else None
        if r.status_code == 401 and status == "invalid_api_key":
            raise SystemExit("the key itself is invalid - copy it again from the ElevenLabs dashboard")
        denied = r.status_code == 401 and status == "missing_permissions"
        # Anything else (200, or a 400/422 validation error) means the request got PAST the
        # permission check, which is all this probe is asking.
        print(f"{scope:<22}{'DENIED' if denied else 'granted':<10}{why}")
        if not denied:
            granted.append(scope)
            if r.status_code == 200:
                if scope == "user_read":
                    subscription = r.json()
                elif scope == "convai_read":
                    agents = r.json().get("agents", [])
                elif scope == "voices_read":
                    voices = r.json().get("voices", [])

    print()
    if subscription:
        print(f"plan: {subscription.get('tier')} | credits used "
              f"{subscription.get('character_count')}/{subscription.get('character_limit')}")
    if agents is not None:
        print(f"agents in workspace: {[(a.get('name'), a.get('agent_id')) for a in agents] or 'none yet'}")
    if voices is not None:
        print(f"voices available: {len(voices)} -> {[v.get('name') for v in voices[:6]]}")
    missing = [s for s, why, *_ in PROBES if why.startswith("NEEDED") and s not in granted]
    print("\nREADY: every needed scope is granted." if not missing
          else f"\nStill missing (needed): {', '.join(missing)}")


if __name__ == "__main__":
    main()
