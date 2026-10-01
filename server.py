"""The whole shop on one port: the web app, the WhatsApp webhook, M-Pesa's callback, and the
endpoint ElevenLabs calls for phone calls.

    python server.py            # http://127.0.0.1:8013

One process, so there is one database, one set of traces and one brain. One port, so a single
cloudflared tunnel serves Meta, Safaricom and ElevenLabs at once.
"""
import contextlib
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

import voice_server  # noqa: E402
import web  # noqa: E402
from brain import Brain  # noqa: E402
from db import init_db  # noqa: E402


def build() -> FastAPI:
    """Everything mounted on one app. The agent is built once and shared by every channel."""
    init_db()
    agent = voice_server.build_server_agent()
    brain = Brain(agent)
    closing = []                                # what to shut down cleanly when the server stops

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        import tools

        for shut in closing:
            with contextlib.suppress(Exception):
                await shut()
        with contextlib.suppress(Exception):
            if tools._MPESA:
                tools._MPESA.close()
        tools.wait_for_emails()                 # a confirmation half-sent is a confirmation lost

    app = FastAPI(title="Mo's shoe shop", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
    # ElevenLabs' Custom LLM endpoint, exactly as it was: /v1/chat/completions and /health.
    app.mount("/elevenlabs", voice_server.create_app(agent=agent))
    try:
        import whatsapp

        wa = whatsapp.router(brain)
        closing.append(wa.graph.aclose)
        app.include_router(wa)
    except Exception as e:                      # no WhatsApp keys yet, or the module is absent
        print(f"  WhatsApp is off: {e}")
    app.include_router(web.router(brain))       # last: it owns "/"
    return app


app = build()

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8013"))
    print(f"\n  Mo's shoe shop on http://127.0.0.1:{port}\n"
          f"  Phone calls: point ElevenLabs at <tunnel>/elevenlabs/v1/chat/completions\n"
          f"  M-Pesa: set MPESA_CALLBACK_URL to <tunnel>/mpesa/callback/$MPESA_CALLBACK_TOKEN\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
