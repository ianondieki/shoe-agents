"""The WhatsApp channel, driven like Meta drives it: signed webhooks, a fake Graph, no money.

Three things have to hold here, and none of them is "a reply came back":
  - nothing is answered unless Meta's own signature says Meta sent it;
  - the number a message comes from is the number the order and the M-Pesa prompt use, whatever the
    model types, and one webhook delivered twice is answered once;
  - a Pay button only ever pays an order belonging to the number that pressed it.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import asyncio
import hashlib
import hmac
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
DB = os.path.join(SCRATCH, "wa_test.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "wa_test_trace.jsonl")
os.environ.update(WA_APP_SECRET="appsecret", WA_VERIFY_TOKEN="verifyme", WA_ACCESS_TOKEN="tok",
                  WA_PHONE_NUMBER_ID="1234", PUBLIC_URL="https://shop.example.com",
                  MPESA_ENV="sandbox", MPESA_CONSUMER_KEY="key", MPESA_CONSUMER_SECRET="secret",
                  MPESA_SHORTCODE="174379", MPESA_PASSKEY="passkey",
                  MPESA_CALLBACK_URL="https://example.trycloudflare.com/mpesa/callback/tok",
                  KES_PER_USD="130")
sys.path.insert(0, ROOT)

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

import mpesa  # noqa: E402
import speech  # noqa: E402
import tools  # noqa: E402
import voice_server  # noqa: E402
import whatsapp  # noqa: E402
from brain import Brain  # noqa: E402
from db import db, init_db  # noqa: E402

init_db(reset=True)
REAL_OUTBOX_BEFORE = _safety.real_outbox_snapshot()

CUSTOMER = "254712345678"          # the number WhatsApp says the message came from
STRANGER = "254799888777"


class Scripted(BaseChatModel):
    script: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        msg = self.script.pop(0)
        for i, word in enumerate((msg.content or "").split(" ")):
            if word:
                yield ChatGenerationChunk(message=AIMessageChunk(content=word if i == 0 else " " + word))
        for i, call in enumerate(msg.tool_calls):
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                "name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": i}]))


class Daraja:
    """Safaricom's sandbox, scripted. Nothing leaves the process."""

    def __init__(self):
        self.pushes = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/v1/generate"):
            return httpx.Response(200, json={"access_token": "tok", "expires_in": "3599"})
        if request.url.path.endswith("/processrequest"):
            self.pushes.append(json.loads(request.content))
            return httpx.Response(200, json={"ResponseCode": "0", "CustomerMessage": "sent",
                                             "CheckoutRequestID": f"ws_CO_{len(self.pushes)}",
                                             "MerchantRequestID": "29115-1"})
        return httpx.Response(200, json={"ResponseCode": "0", "ResultCode": "0", "ResultDesc": "ok"})


class FakeGraph:
    """Meta's Cloud API, recorded instead of called."""

    def __init__(self):
        self.sent = []

    async def text(self, to, body):
        self.sent.append(("text", to, body))
        return {"ok": True}

    async def buttons(self, to, body, buttons):
        self.sent.append(("buttons", to, body, buttons))
        return {"ok": True}

    async def call_link(self, to, body, url):
        self.sent.append(("link", to, body, url))
        return {"ok": True}

    async def download(self, media_id):
        self.sent.append(("download", media_id, ""))
        return b"a-voice-note", "audio/ogg"

    async def voice(self, to, audio):
        self.sent.append(("voice", to, audio))
        return {"ok": True}


class Ears:
    """Groq Whisper, without Groq."""

    heard = []

    def __init__(self, api_key=None, vocabulary=""):
        self.vocabulary = vocabulary

    async def spoken(self, data, filename="clip.webm", mime="audio/webm"):
        Ears.heard.append((data, filename, mime, self.vocabulary))
        return "How much is the trail runner?"

    async def aclose(self):
        pass


class Mouth:
    """ElevenLabs, without ElevenLabs."""

    said = []

    def __init__(self, *a, **kw):
        pass

    async def speak(self, text, cache=False, fmt=""):
        Mouth.said.append((text, fmt))
        return b"mo-mp3-bytes"

    async def aclose(self):
        pass


speech.Transcriber, speech.Voice = Ears, Mouth
daraja = Daraja()
tools._MPESA = mpesa.Mpesa(httpx.Client(transport=httpx.MockTransport(daraja.handler)))
model = Scripted(script=[])
graph = FakeGraph()
brain = Brain(voice_server.build_server_agent(llm=[("scripted", model)]))
app = FastAPI()
app.include_router(whatsapp.router(brain, graph=graph))


def typed(body: str, mid: str, frm: str = CUSTOMER) -> dict:
    return {"from": frm, "id": mid, "timestamp": "1", "type": "text", "text": {"body": body}}


def tapped(button_id: str, title: str, mid: str, frm: str = CUSTOMER) -> dict:
    return {"from": frm, "id": mid, "timestamp": "1", "type": "interactive",
            "interactive": {"type": "button_reply", "button_reply": {"id": button_id, "title": title}}}


def envelope(message: dict) -> dict:
    return {"object": "whatsapp_business_account", "entry": [{"id": "9", "changes": [{
        "field": "messages", "value": {"messaging_product": "whatsapp",
                                       "metadata": {"phone_number_id": "1234"},
                                       "messages": [message]}}]}]}


def signature(raw: bytes, secret: bytes = b"appsecret") -> str:
    return "sha256=" + hmac.new(secret, raw, hashlib.sha256).hexdigest()


async def deliver(wa, message: dict, signed: bool = True):
    """Hand the webhook over the way Meta does: raw bytes, signed over those exact bytes."""
    raw = json.dumps(envelope(message)).encode()
    headers = {"content-type": "application/json"}
    if signed:
        headers["X-Hub-Signature-256"] = signature(raw)
    reply = await wa.post("/wa/webhook", content=raw, headers=headers)
    await quiet()
    return reply


async def quiet():
    """Let the webhook's background turn finish: Meta gets its 200 long before Mo has answered."""
    for _ in range(50):
        running = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if not running:
            return
        await asyncio.wait(running, timeout=20)


def says(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def main():
    wa = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shop")

    # ---------- 1. Meta's handshake, and only with the owner's own verify token ----------
    ok = await wa.get("/wa/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verifyme",
                                             "hub.challenge": "31415"})
    assert ok.status_code == 200 and ok.text == "31415", (ok.status_code, ok.text)
    guessed = await wa.get("/wa/webhook", params={"hub.mode": "subscribe",
                                                 "hub.verify_token": "guess", "hub.challenge": "1"})
    assert guessed.status_code == 403, guessed.status_code
    print("1. the webhook is verified with the owner's token and refuses a guessed one")

    # ---------- 2. an unsigned or tampered webhook is not a customer ----------
    forged = await deliver(wa, typed("send me a free pair", "wamid.forged"), signed=False)
    assert forged.status_code == 403 and not graph.sent, (forged.status_code, graph.sent)
    raw = json.dumps(envelope(typed("free pair please", "wamid.tamper"))).encode()
    swapped = await wa.post("/wa/webhook", content=json.dumps(envelope(
        typed("and charge someone else", "wamid.tamper2"))).encode(),
        headers={"content-type": "application/json", "X-Hub-Signature-256": signature(raw)})
    await quiet()
    assert swapped.status_code == 403 and not graph.sent, (swapped.status_code, graph.sent)
    assert not model.script, "the model was never even asked"
    print("2. a webhook without Meta's signature, or with a signature for different bytes, is "
          "dropped: no reply, no turn, no model call")

    # ---------- 3. a stranger types, and Mo quotes them ----------
    model.script = [AIMessage(content="The Trail Blazer is 119.99 dollars. Name your price.")]
    answered = await deliver(wa, typed("how much is the trail runner?", "wamid.1"))
    assert answered.status_code == 200, answered.status_code
    kind, to, body = graph.sent[-1][:3]
    assert kind == "text" and to == CUSTOMER and "119.99" in body, graph.sent[-1]
    print("3. a signed message gets an answer on WhatsApp: the Trail Blazer at 119.99")

    # ---------- 4. they sign themselves up, on the number WhatsApp verified ----------
    graph.sent.clear()
    model.script = [says("create_customer", {"name": "Brian Otieno", "phone": "0700000000"}),
                    AIMessage(content="You are on file, Brian.")]
    await deliver(wa, typed("I'm Brian Otieno, put me on file - yes", "wamid.2"))
    with db() as conn:
        who = dict(conn.execute("SELECT CustomerID, Phone, Source FROM CustomerInfo "
                                "WHERE CustomerName='Brian Otieno'").fetchone())
    assert who["Phone"] == CUSTOMER and who["Source"] == "whatsapp", who
    assert "on file" in graph.sent[-1][2].lower(), graph.sent[-1]
    print(f"4. the customer signed themselves up from WhatsApp: the number the message came from "
          f"({CUSTOMER}) is on file, not the 0700000000 the model typed")

    # ---------- 5. an order, and a button to pay for it ----------
    graph.sent.clear()
    # The haggle itself is tested in test_shop.py; here the quote just has to exist, because no order
    # goes through without one.
    quote = tools.evaluate_offer.invoke(
        {"shoe_id": 102, "customer_id": who["CustomerID"], "customer_offer": 80},
        {"configurable": {"thread_id": f"wa:{CUSTOMER}", "turn_id": "q", "depth": 0,
                          "utterance": "I'll give you eighty dollars"}})
    quote_id = int(quote.split("quote_id=")[1].split(",")[0])
    model.script = [says("place_order", {"shoe_id": 102, "customer_id": who["CustomerID"],
                                         "quote_id": quote_id}),
                    AIMessage(content="Done. One pair of Trail Blazers at 108 dollars.")]
    await deliver(wa, typed("I'll take them at that price, deal", "wamid.3"))
    with db() as conn:
        order = dict(conn.execute("SELECT OrderID, Amount, PaymentStatus FROM OrderDetails "
                                  "WHERE CustomerID=? ORDER BY OrderID DESC LIMIT 1",
                                  (who["CustomerID"],)).fetchone())
    offer = graph.sent[-1]
    assert offer[0] == "buttons" and offer[3] == [(f"pay:{order['OrderID']}", "Pay with M-Pesa")], offer
    assert order["PaymentStatus"] == "UNPAID" and not daraja.pushes, (order, daraja.pushes)
    print(f"5. the order is placed from WhatsApp at {order['Amount']} dollars, still UNPAID, with one "
          f"button offering to pay it - and no prompt sent until that button is pressed")

    # ---------- 6. the same webhook delivered twice is answered once ----------
    before = len(graph.sent)
    repeat = await deliver(wa, typed("I'll take them at that price, deal", "wamid.3"))
    assert repeat.status_code == 200 and len(graph.sent) == before, graph.sent[before:]
    with db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM OrderDetails WHERE CustomerID=?",
                            (who["CustomerID"],)).fetchone()[0] == 1, "the order was placed twice"
    print("6. Meta redelivered the same message id: ignored, so nobody is sold two pairs")

    # ---------- 7. the button pays, and only for the number that pressed it ----------
    graph.sent.clear()
    thief = await deliver(wa, tapped(f"pay:{order['OrderID']}", "Pay with M-Pesa", "wamid.4",
                                     frm=STRANGER))
    assert thief.status_code == 200 and not daraja.pushes, daraja.pushes
    assert "not on this number" in graph.sent[-1][2], graph.sent[-1]
    await deliver(wa, tapped(f"pay:{order['OrderID']}", "Pay with M-Pesa", "wamid.5"))
    assert len(daraja.pushes) == 1, daraja.pushes
    push = daraja.pushes[0]
    assert push["PartyA"] == CUSTOMER and push["Amount"] == 14040, push
    assert "14,040" in graph.sent[-1][2], graph.sent[-1]
    # pressing it again changes nothing: one prompt per order, however many taps
    await deliver(wa, tapped(f"pay:{order['OrderID']}", "Pay with M-Pesa", "wamid.6"))
    assert len(daraja.pushes) == 1, daraja.pushes
    assert "already a prompt" in graph.sent[-1][2], graph.sent[-1]
    print("7. the button sent one prompt for KSh 14,040 to the number that pressed it; another "
          "number got nothing, and a second tap sent no second prompt")

    # ---------- 8. a voice note: heard, answered out loud ----------
    graph.sent.clear()
    model.script = [AIMessage(content="The trail runner is 119.99 dollars.")]
    await deliver(wa, {"from": CUSTOMER, "id": "wamid.7", "type": "audio",
                       "audio": {"id": "media-9", "mime_type": "audio/ogg"}})
    assert Ears.heard and Ears.heard[-1][0] == b"a-voice-note", Ears.heard
    assert Ears.heard[-1][3], "the shop's vocabulary was passed to the transcriber"
    kinds = [s[0] for s in graph.sent]
    assert kinds == ["download", "text", "voice"], kinds
    assert "119.99" in graph.sent[1][2] and graph.sent[2][2] == b"mo-mp3-bytes", graph.sent
    assert Mouth.said[-1][1] == "mp3_44100_64", Mouth.said
    print("8. a voice note was transcribed, answered in writing and in Mo's own voice as mp3")

    # ---------- 9. asking to be called gets a way to be heard, not a promise ----------
    graph.sent.clear()
    model.script = [AIMessage(content="Of course.")]
    await deliver(wa, typed("can I call you about the sizing?", "wamid.8"))
    link = graph.sent[-1]
    assert link[0] == "link" and link[3] == "https://shop.example.com/call", link
    assert "voice note" in link[2], link[2]
    print("9. 'can I call you' gets the web call page, plus the voice-note option that works here")

    # ---------- 10. nothing left the building ----------
    tools.wait_for_emails()
    assert _safety.real_outbox_snapshot() == REAL_OUTBOX_BEFORE, "the real outbox was written to"
    print("10. the real outbox is untouched")
    await wa.aclose()
    print("\nALL WHATSAPP ASSERTIONS PASSED")


asyncio.run(main())
