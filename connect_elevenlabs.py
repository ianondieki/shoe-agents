"""Point the ElevenLabs agent at this machine's voice server, through the current tunnel.

A free Cloudflare quick tunnel gets a new random address every time it starts, so the agent's
Custom LLM URL goes stale on every restart. Run this after starting the server and the tunnel:

    python voice_server.py                                                  (terminal 1)
    tools\\cloudflared.exe tunnel --url http://127.0.0.1:8013 --protocol quic (terminal 2)
    python connect_elevenlabs.py                                            (terminal 3)

It finds the tunnel address, checks the whole path works from the outside, then updates the agent.
Pass --url https://... to use a tunnel it cannot detect (ngrok, a fixed domain).

--protocol quic matters on this machine: antivirus HTTPS scanning (Avast) breaks the TLS connection
that tunnel tools make to their control servers; QUIC runs over UDP and is not intercepted.
"""
import argparse
import os
import sys

try:
    # See agent_core.py: antivirus/proxy TLS interception breaks certifi-based verification.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API = "https://api.elevenlabs.io"
LOCAL = f"http://127.0.0.1:{os.getenv('VOICE_PORT', '8013')}"


def tunnel_url() -> str | None:
    """cloudflared serves its quick-tunnel hostname on its metrics port (20241-20245)."""
    for port in range(20241, 20246):
        try:
            host = httpx.get(f"http://127.0.0.1:{port}/quicktunnel", timeout=2).json().get("hostname")
            if host:
                return f"https://{host}"
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", help="public base URL of the tunnel, if it cannot be detected")
    args = ap.parse_args()

    key, agent_id = os.getenv("ELEVENLABS_API_KEY"), os.getenv("ELEVENLABS_AGENT_ID")
    if not key or not agent_id:
        sys.exit("set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in .env")

    # 1. the server, locally
    try:
        httpx.get(f"{LOCAL}/health", timeout=3).raise_for_status()
    except Exception:
        sys.exit(f"voice_server is not answering on {LOCAL} - start it first: python voice_server.py")
    print(f"✓ voice server up on {LOCAL}")

    # 2. the tunnel
    url = (args.url or tunnel_url() or "").rstrip("/")
    if not url:
        sys.exit("no tunnel found - start one: tools\\cloudflared.exe tunnel --url "
                 f"{LOCAL} --protocol quic   (or pass --url)")
    print(f"✓ tunnel at {url}")

    # 3. the whole path, from the public side: reachable, and locked
    try:
        httpx.get(f"{url}/health", timeout=20).raise_for_status()
        denied = httpx.post(f"{url}/v1/chat/completions", json={"messages": []}, timeout=20).status_code
    except Exception as e:
        sys.exit(f"the tunnel does not reach the server yet ({type(e).__name__}) - wait a few seconds, retry")
    if denied != 401 and os.getenv("VOICE_SHARED_SECRET"):
        sys.exit(f"the public endpoint answered {denied} without the secret - is it the right server?")
    print("✓ reachable through the tunnel, and refuses requests without the shared secret")

    # 4. the agent
    headers = {"xi-api-key": key}
    agent = httpx.get(f"{API}/v1/convai/agents/{agent_id}", headers=headers, timeout=30)
    agent.raise_for_status()
    prompt = agent.json()["conversation_config"]["agent"]["prompt"]
    custom = dict(prompt.get("custom_llm") or {})
    if custom.get("url") == f"{url}/v1":
        print(f"✓ agent already points at {url}/v1 - nothing to change")
        return
    custom["url"] = f"{url}/v1"
    r = httpx.patch(f"{API}/v1/convai/agents/{agent_id}", headers=headers, timeout=30,
                    json={"conversation_config": {"agent": {"prompt": {"llm": "custom-llm", "custom_llm": custom}}}})
    r.raise_for_status()
    print(f"✓ agent {agent.json().get('name')!r} now uses {url}/v1")


if __name__ == "__main__":
    main()
