"""List the models each key can actually use, and flag what .env points at.

Run this when a call fails with 403 tier_not_allowed, or 404 for a retired model ID:

    python check_models.py

'->' marks the model .env selects. Mistral reports per-model capabilities, so models that
cannot do tool calling are left out; Groq does not report them, so its list is unfiltered.
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

load_dotenv()

PROVIDERS = [
    ("Mistral", "https://api.mistral.ai/v1/models", "MISTRAL_API_KEY", "MISTRAL_MODEL"),
    ("Groq", "https://api.groq.com/openai/v1/models", "GROQ_API_KEY", "GROQ_MODEL"),
]


def tool_capable(model: dict) -> bool:
    """False only when the provider says so; unknown counts as usable."""
    caps = model.get("capabilities")
    return caps.get("function_calling", True) if isinstance(caps, dict) else True


for label, url, key_var, model_var in PROVIDERS:
    key, configured = os.getenv(key_var), os.getenv(model_var)
    print(f"\n=== {label}: {model_var}={configured or 'unset (using the code default)'} ===")
    if not key:
        print(f"  {key_var} is not set in .env")
        continue
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  could not list models: {type(e).__name__}: {e}")
        continue

    models = r.json().get("data", [])
    every_id = {m["id"] for m in models}
    usable = sorted({m["id"] for m in models if tool_capable(m)})
    for mid in usable:
        print(f"  {'->' if mid == configured else '  '} {mid}")
    if configured and configured not in every_id:
        print(f"  ⚠️  {configured} is not available to this key — every call to it fails")
    elif configured and configured not in usable:
        print(f"  ⚠️  {configured} cannot do tool calling, which this agent needs")
