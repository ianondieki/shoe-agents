"""Mo on WhatsApp: typed messages, voice notes, and a button that pays.

Meta's WhatsApp Cloud API, used the free way: we only ever answer somebody who messaged us first,
which stays inside the free 24-hour service window and needs no paid template.

    a customer types        -> Mo -> text back
    a customer sends a note -> Groq Whisper -> Mo -> a voice note back in Mo's own voice
    an order is placed      -> a "Pay with M-Pesa" button -> the STK prompt on that same phone

The number a message comes from is verified by WhatsApp itself, so it is the one thing here that can
be trusted: it recognises a returning customer, and it is the number create_customer uses for a new
one. Nobody can order on somebody else's number by typing it.

A voice *call* placed inside WhatsApp needs Meta's Business Calling API, which is not available on a
free test number. So the voice-note path above is how you talk to Mo here, and a "Talk to Mo" button
opens the web call page, where the same brain answers out loud in real time.

In .env (all free to obtain):
    WA_VERIFY_TOKEN=...      # any random string; type the same one into Meta's webhook form
    WA_APP_SECRET=...        # App settings -> Basic -> App secret. It signs every webhook.
    WA_ACCESS_TOKEN=...      # a token with whatsapp_business_messaging
    WA_PHONE_NUMBER_ID=...   # WhatsApp -> API setup
    PUBLIC_URL=https://...   # where Meta can reach this server (the cloudflared tunnel)
"""
import asyncio
import hashlib
import hmac
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import tools
from brain import Brain, customer_by_phone
from db import db

GRAPH = os.getenv("WA_GRAPH", "https://graph.facebook.com/v21.0")
# Answering happens after Meta has its 200, in a task of its own. The event loop holds tasks only
# weakly, so a task nobody keeps a reference to can be collected while it waits on the model - and
# the customer never hears back.
_ANSWERING: set = set()


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def configured() -> bool:
    return bool(env("WA_ACCESS_TOKEN") and env("WA_PHONE_NUMBER_ID") and env("WA_APP_SECRET"))


def signed_by_meta(raw: bytes, header: str) -> bool:
    """Every webhook carries an HMAC of its exact bytes, keyed by the app secret. With no secret in
    .env there is no way to tell Meta from anyone else who found the URL, so nothing is accepted."""
    secret = env("WA_APP_SECRET")
    if not secret or not header.startswith("sha256="):
        return False
    want = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, header.split("=", 1)[1])


def already_seen(message_id: str) -> bool:
    """Meta resends a webhook until it gets a 200, and answering the same message twice could put a
    second payment prompt on somebody's phone."""
    if not message_id:
        return False
    with db() as conn:
        if conn.execute("SELECT 1 FROM WaSeen WHERE MessageID=?", (message_id,)).fetchone():
            return True
        conn.execute("INSERT INTO WaSeen (MessageID, SeenAt) VALUES (?, datetime('now'))",
                     (message_id,))
        conn.execute("DELETE FROM WaSeen WHERE SeenAt < datetime('now', '-2 days')")
    return False


def owns_order(customer_id: int | None, order_id: int) -> bool:
    """Whether that order belongs to this customer. The customer comes from the number WhatsApp
    verified, never from the button: otherwise a guessed button id would be enough to pay for
    somebody else's order, or to make them pay for yours."""
    if not customer_id:
        return False
    with db() as conn:
        return bool(conn.execute("SELECT 1 FROM OrderDetails WHERE OrderID=? AND CustomerID=?",
                                 (order_id, customer_id)).fetchone())


class Graph:
    """The handful of Cloud API calls this shop makes. Replaced by a fake in the tests."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=30)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {env('WA_ACCESS_TOKEN')}"}

    async def _post(self, path: str, **kw) -> dict:
        r = await self.client.post(f"{GRAPH}/{path}", headers=self.headers, **kw)
        if r.status_code >= 400:
            # Never raise inside a webhook: a failed send must not make Meta redeliver the message.
            print(f"  WhatsApp refused {path}: {r.status_code} {r.text[:200]}")
            return {}
        return r.json()

    async def text(self, to: str, body: str) -> dict:
        return await self._post(f"{env('WA_PHONE_NUMBER_ID')}/messages", json={
            "messaging_product": "whatsapp", "to": to, "type": "text",
            "text": {"preview_url": False, "body": body[:4000]}})

    async def buttons(self, to: str, body: str, buttons: list[tuple[str, str]]) -> dict:
        """Up to three tappable replies, as (id, title). WhatsApp allows 20 characters of title."""
        return await self._post(f"{env('WA_PHONE_NUMBER_ID')}/messages", json={
            "messaging_product": "whatsapp", "to": to, "type": "interactive",
            "interactive": {"type": "button", "body": {"text": body[:1000]},
                            "action": {"buttons": [
                                {"type": "reply", "reply": {"id": i, "title": t[:20]}}
                                for i, t in buttons]}}})

    async def call_link(self, to: str, body: str, url: str) -> dict:
        """A button that opens the web call page, the nearest thing to phoning Mo from in here."""
        return await self._post(f"{env('WA_PHONE_NUMBER_ID')}/messages", json={
            "messaging_product": "whatsapp", "to": to, "type": "interactive",
            "interactive": {"type": "cta_url", "body": {"text": body[:1000]},
                            "action": {"name": "cta_url",
                                       "parameters": {"display_text": "Talk to Mo", "url": url}}}})

    async def download(self, media_id: str) -> tuple[bytes, str]:
        """A voice note arrives as an id: ask where it is, then fetch it with the same token."""
        where = await self.client.get(f"{GRAPH}/{media_id}", headers=self.headers)
        where.raise_for_status()
        found = where.json()
        got = await self.client.get(found["url"], headers=self.headers)
        got.raise_for_status()
        return got.content, found.get("mime_type", "audio/ogg")

    async def voice(self, to: str, audio: bytes) -> dict:
        """Mo's own voice as a voice note: upload the mp3, then send what the upload gave back."""
        up = await self._post(f"{env('WA_PHONE_NUMBER_ID')}/media",
                              data={"messaging_product": "whatsapp", "type": "audio/mpeg"},
                              files={"file": ("mo.mp3", audio, "audio/mpeg")})
        if not up.get("id"):
            return {}
        return await self._post(f"{env('WA_PHONE_NUMBER_ID')}/messages", json={
            "messaging_product": "whatsapp", "to": to, "type": "audio", "audio": {"id": up["id"]}})

    async def aclose(self):
        await self.client.aclose()


def router(brain: Brain, graph: Graph | None = None) -> APIRouter:
    """The two routes Meta needs. Raises when the keys are missing, which is how server.py decides
    whether this channel is on."""
    if graph is None and not configured():
        raise RuntimeError("no WA_ACCESS_TOKEN / WA_PHONE_NUMBER_ID / WA_APP_SECRET in .env")
    api = APIRouter()
    graph = graph or Graph()
    api.graph = graph                               # so the server can close its client on shutdown

    @api.get("/wa/webhook", response_class=PlainTextResponse)
    async def verify(request: Request):
        """Meta's one-time handshake when the webhook URL is saved."""
        asked = request.query_params
        if (env("WA_VERIFY_TOKEN") and asked.get("hub.mode") == "subscribe"
                and hmac.compare_digest(asked.get("hub.verify_token", ""), env("WA_VERIFY_TOKEN"))):
            return PlainTextResponse(asked.get("hub.challenge", ""))
        return PlainTextResponse("no", status_code=403)

    @api.post("/wa/webhook")
    async def incoming(request: Request):
        """A message from a customer. Answered in the background, because Meta wants its 200 within
        seconds and a turn with the model takes longer than that."""
        raw = await request.body()
        if not signed_by_meta(raw, request.headers.get("x-hub-signature-256", "")):
            return JSONResponse({"ok": False}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": True})
        if not isinstance(body, dict):              # valid JSON, but not a webhook: nothing to do
            return JSONResponse({"ok": True})
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                for message in (change.get("value") or {}).get("messages", []):
                    answering = asyncio.create_task(handle(brain, graph, message))
                    _ANSWERING.add(answering)
                    answering.add_done_callback(_ANSWERING.discard)
        return JSONResponse({"ok": True})

    return api


async def handle(brain: Brain, graph: Graph, message: dict):
    """One WhatsApp message, start to finish. Never raises: a webhook that errors is redelivered."""
    wa_id = str(message.get("from") or "")
    try:
        if not wa_id or already_seen(str(message.get("id") or "")):
            return
        # Who this number is, if anyone. Everything else - the order, the button, the prompt - is
        # checked against this and never against what the message says.
        known = customer_by_phone(wa_id) or {"phone": wa_id}
        said, by_voice = await heard(graph, wa_id, known, message)
        if said is None:                               # a button press, or nothing we can answer
            return
        if not said:
            await graph.text(wa_id, "I did not catch that. Type it, or send the note again?")
            return

        lines, meta = [], {}
        async for kind, payload in brain.turn(f"wa:{wa_id}", said, known=known, channel="whatsapp"):
            if kind == "say":
                lines.append(payload)
            else:
                meta = payload
        answer = " ".join(lines).strip() or "Say that again?"

        if by_voice:
            await speak(graph, wa_id, answer)
        else:
            await graph.text(wa_id, answer)

        if meta.get("ordered") and meta.get("order_id"):
            await offer_payment(graph, wa_id, known, meta["order_id"])
        elif wants_a_call(said):
            await call_me_back(graph, wa_id)
    except Exception as e:                             # a dropped reply beats a redelivery storm
        print(f"  WhatsApp turn failed for {wa_id}: {type(e).__name__}: {e}")


async def heard(graph: Graph, wa_id: str, known: dict, message: dict) -> tuple[str | None, bool]:
    """What the customer said, and whether they said it out loud.

    Returns (None, _) when there is nothing for the brain: a button press is handled here, and a
    sticker or a photo gets a short answer rather than a turn.
    """
    kind = message.get("type")
    if kind == "text":
        return ((message.get("text") or {}).get("body") or "").strip(), False
    if kind == "interactive":
        return await pressed(graph, wa_id, known, message), False
    if kind == "button":                               # a template's quick reply
        return ((message.get("button") or {}).get("text") or "").strip(), False
    if kind in ("audio", "voice"):
        return await listened(graph, message), True
    await graph.text(wa_id, "I can read a message or listen to a voice note. Both get you a price.")
    return None, False


async def pressed(graph: Graph, wa_id: str, known: dict, message: dict) -> str | None:
    """A tapped button. "pay:7" takes payment; anything else is treated as the words on it."""
    reply = ((message.get("interactive") or {}).get("button_reply")
             or (message.get("interactive") or {}).get("list_reply") or {})
    pressed_id = str(reply.get("id") or "")
    if pressed_id.startswith("pay:"):
        await pay(graph, wa_id, known, int(pressed_id.split(":", 1)[1]))
        return None
    if pressed_id == "call":
        await call_me_back(graph, wa_id)
        return None
    return (reply.get("title") or "").strip()


async def listened(graph: Graph, message: dict) -> str:
    """A voice note, turned into words by the same ear that hears phone calls."""
    import speech

    media = (message.get("audio") or message.get("voice") or {}).get("id")
    if not media:
        return ""
    data, mime = await graph.download(media)
    ears = speech.Transcriber(vocabulary=speech.shop_vocabulary())
    try:
        return await ears.spoken(data, "note.ogg", mime)
    except speech.SpeechUnavailable as e:
        print(f"  WhatsApp could not transcribe: {e}")
        return ""
    finally:
        await ears.aclose()


async def speak(graph: Graph, wa_id: str, answer: str):
    """Answer a voice note with a voice note, and in text too, so the price can be read back later."""
    import speech

    await graph.text(wa_id, answer)
    voice = speech.Voice()
    try:
        await graph.voice(wa_id, await voice.speak(answer[:600], fmt="mp3_44100_64"))
    except speech.SpeechUnavailable as e:
        print(f"  Mo's voice is unavailable on WhatsApp: {e}")   # the text above still went out
    finally:
        await voice.aclose()


async def offer_payment(graph: Graph, wa_id: str, known: dict, order_id: int):
    if not owns_order(known.get("customer_id"), order_id):
        return
    await graph.buttons(wa_id, "Pay now and those are yours. The prompt comes to this number.",
                        [(f"pay:{order_id}", "Pay with M-Pesa")])


async def pay(graph: Graph, wa_id: str, known: dict, order_id: int):
    """The Pay button. The prompt goes to the number that pressed it and to no other."""
    if not owns_order(known.get("customer_id"), order_id):
        await graph.text(wa_id, "That order is not on this number.")
        return
    out = await asyncio.to_thread(tools.request_push, order_id, wa_id)
    if out.get("sent"):
        await graph.text(wa_id, f"Check your phone: a prompt for KSh {out['amount_kes']:,} is on its "
                                f"way. Enter your M-Pesa PIN and I will confirm here.")
    elif out.get("waiting"):
        await graph.text(wa_id, "There is already a prompt on your phone for this order. Finish that "
                                "one, or let it expire and tap again.")
    else:
        await graph.text(wa_id, f"The prompt did not go out: {out.get('error', 'M-Pesa is quiet')}. "
                                f"Your order is held either way.")


def wants_a_call(said: str) -> bool:
    words = said.lower()
    return any(w in words for w in ("call me", "call you", "phone call", "ring me", "can i call",
                                    "talk to you", "speak to you"))


async def call_me_back(graph: Graph, wa_id: str):
    """WhatsApp calling needs Meta's Business Calling API, which a free test number does not have.
    What we do have is the same brain answering out loud on the web."""
    where = env("PUBLIC_URL").rstrip("/")
    if not where:
        await graph.text(wa_id, "Send me a voice note instead and I will answer out loud.")
        return
    await graph.call_link(wa_id, "I cannot take a WhatsApp call on this number yet. Tap below and we "
                                 "talk out loud in the browser - or just send a voice note here.",
                          f"{where}/call")
