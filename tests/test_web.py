"""The web shop, driven like a browser: no network, no money, no real email.

What matters here is not that pages render. It is that the web cannot do anything the phone agent is
not allowed to do: the price comes from the shelf and never from the form, a cookie cannot pay for
somebody else's order, and an order is PAID only once Safaricom's own API says so.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import asyncio
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
DB = os.path.join(SCRATCH, "web_test.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "web_test_trace.jsonl")
os.environ.update(WEB_SECRET="test-secret", MPESA_ENV="sandbox", MPESA_CONSUMER_KEY="key",
                  MPESA_CONSUMER_SECRET="secret", MPESA_SHORTCODE="174379", MPESA_PASSKEY="passkey",
                  MPESA_CALLBACK_URL="https://example.trycloudflare.com/mpesa/callback/tok",
                  MPESA_CALLBACK_TOKEN="tok", KES_PER_USD="130")
sys.path.insert(0, ROOT)

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

import mpesa  # noqa: E402
import tools  # noqa: E402
import voice_server  # noqa: E402
import web  # noqa: E402
from brain import Brain  # noqa: E402
from db import db, init_db  # noqa: E402

init_db(reset=True)
REAL_OUTBOX_BEFORE = _safety.real_outbox_snapshot()


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
    """Safaricom's sandbox, scripted."""

    def __init__(self):
        self.outcome = {"ResponseCode": "0", "ResultCode": "0", "ResultDesc": "ok"}
        self.pushes = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/v1/generate"):
            return httpx.Response(200, json={"access_token": "tok", "expires_in": "3599"})
        if request.url.path.endswith("/processrequest"):
            self.pushes += 1
            return httpx.Response(200, json={"ResponseCode": "0", "CustomerMessage": "sent",
                                             "CheckoutRequestID": f"ws_CO_{self.pushes}",
                                             "MerchantRequestID": "29115-1"})
        return httpx.Response(200, json=self.outcome)


daraja = Daraja()
tools._MPESA = mpesa.Mpesa(httpx.Client(transport=httpx.MockTransport(daraja.handler)))
model = Scripted(script=[])
app = FastAPI()
app.include_router(web.router(Brain(voice_server.build_server_agent(llm=[("scripted", model)]))))


def csrf_from(html: str) -> str:
    return re.search(r'name="csrf" value="([a-f0-9]+)"', html).group(1)


def sse(text: str) -> list[dict]:
    return [json.loads(part[6:]) for part in text.split("\n\n")
            if part.startswith("data: ") and not part.strip().endswith("[DONE]")]


async def main():
    shop = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shop",
                             follow_redirects=False)
    other = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shop",
                              follow_redirects=False)

    # ---------- 1. the shop opens, and the browser gets a session of its own ----------
    front = await shop.get("/")
    assert front.status_code == 200 and "set-cookie" in front.headers
    assert front.text.count('class="pair"') == 4, "four pairs are in stock in the seed shop"
    assert "KSh" in front.text and "Name your price." in front.text
    csrf = csrf_from(front.text)
    print("1. the shop: 4 pairs in stock, each with a dollar tag and a shilling price, and the "
          "browser has a signed session")

    # ---------- 2. nobody orders anonymously, and the form cannot set a price ----------
    sent_away = await shop.post("/buy", data={"shoe_id": "102", "csrf": csrf})
    assert sent_away.headers["location"].startswith("/profile?next=/buy"), sent_away.headers
    no_token = await shop.post("/buy", data={"shoe_id": "102", "csrf": "wrong"})
    assert no_token.headers["location"].startswith("/?err="), no_token.headers
    print("2. buying: an unknown browser signs up first, and a post without the form's token is refused")

    # ---------- 3. signing yourself up, from the web, with no password ----------
    made = await shop.post("/profile", data={"name": "Grace Wanjiru", "phone": "0712345678",
                                             "csrf": csrf, "next": "/buy", "shoe_id": "102"})
    assert "Finish the order" in made.text, made.text[:200]
    with db() as conn:
        mine = dict(conn.execute("SELECT * FROM CustomerInfo WHERE Phone='254712345678'").fetchone())
    assert mine["Source"] == "web" and mine["CreatedAt"], mine
    bad = await shop.post("/profile", data={"name": "Grace", "phone": "12345", "csrf": csrf})
    assert "Kenyan mobile number" in bad.text, "a nonsense number was accepted"
    print(f"3. profiles: 0712345678 became customer {mine['CustomerID']} (Source=web); '12345' is "
          f"refused with what to fix")

    # ---------- 4. the order is priced from the shelf, by the tool the phone agent uses ----------
    bought = await shop.post("/buy", data={"shoe_id": "102", "csrf": csrf, "amount": "1"})
    where = bought.headers["location"]
    order_id = int(where.rsplit("/", 1)[1])
    with db() as conn:
        order = dict(conn.execute("SELECT * FROM OrderDetails WHERE OrderID=?", (order_id,)).fetchone())
    assert order["Amount"] == 119.99 and order["PaymentStatus"] == "UNPAID", order
    assert order["CustomerID"] == mine["CustomerID"]
    page = await shop.get(where)
    assert "KSh 15,599" in page.text and "Pay KSh" in page.text, "the receipt prices in shillings"
    print(f"4. the order: 'amount=1' in the form changed nothing - order {order_id} is 119.99 dollars, "
          f"KSh 15,599, taken from the shelf")

    # ---------- 5. another browser cannot see or pay that order ----------
    theirs = await other.get("/")
    assert "not placed from here" in (await other.get(f"/orders/{order_id}")).text
    assert (await other.get(f"/api/orders/{order_id}/status")).status_code == 404
    stolen = await other.post(f"/orders/{order_id}/pay", data={"csrf": csrf_from(theirs.text)})
    assert stolen.headers["location"] == "/orders", stolen.headers
    assert daraja.pushes == 0, "a stranger's click sent an M-Pesa prompt"
    print("5. somebody else's order: hidden, unpayable, and no prompt went out")

    # ---------- 6. paying, and what it takes to be believed ----------
    paying = await shop.post(f"/orders/{order_id}/pay", data={"csrf": csrf})
    assert paying.headers["location"].endswith("?sent=1"), paying.headers
    assert daraja.pushes == 1
    with db() as conn:
        waiting = dict(conn.execute("SELECT * FROM Payment WHERE OrderID=?", (order_id,)).fetchone())
    assert waiting["Status"] == "PENDING" and waiting["AmountKes"] == 15599, waiting
    daraja.outcome = {"ResponseCode": "0", "errorMessage": "transaction is being processed"}
    assert (await shop.get(f"/api/orders/{order_id}/status")).json()["status"] == "PENDING"

    forged = await shop.post("/mpesa/callback/wrong", json={"Body": {"stkCallback": {
        "CheckoutRequestID": waiting["CheckoutRequestID"], "ResultCode": 0,
        "CallbackMetadata": {"Item": [{"Name": "MpesaReceiptNumber", "Value": "FAKE"}]}}}})
    assert forged.json()["ResultDesc"] == "ignored", forged.json()
    with db() as conn:
        assert conn.execute("SELECT PaymentStatus FROM OrderDetails WHERE OrderID=?",
                            (order_id,)).fetchone()["PaymentStatus"] == "PENDING"

    daraja.outcome = {"ResponseCode": "0", "ResultCode": "0", "ResultDesc": "ok"}
    assert (await shop.get(f"/api/orders/{order_id}/status")).json()["status"] == "PAID"
    receipt = await shop.get(f"/orders/{order_id}")
    assert "Paid" in receipt.text and "Pay KSh" not in receipt.text
    print("6. paying: one prompt for KSh 15,599, a callback with the wrong token ignored, and PAID "
          "only once Safaricom's own API confirmed it")

    # ---------- 7. haggling in the browser, then paying the haggled price ----------
    model.script = [
        AIMessage(content="", tool_calls=[{"name": "customer_and_stock", "id": "c1", "type": "tool_call",
                                           "args": {"customer_name": "Grace Wanjiru",
                                                    "activity": "trail"}}]),
        AIMessage(content="The cushioned trail runner is 120 dollars."),
    ]
    talked = sse((await shop.post("/api/chat", json={"text": "how much is the trail runner?"})).text)
    said = [line["say"] for line in talked if "say" in line]
    assert said and said[0] == voice_server.FILLER["customer_and_stock"], said
    assert any("120 dollars" in s for s in said), said
    assert talked[-1]["done"]["ordered"] is False, talked[-1]

    quote = tools.evaluate_offer.invoke(
        {"shoe_id": 102, "customer_id": mine["CustomerID"], "customer_offer": 80},
        {"configurable": {"thread_id": "web:haggle", "turn_id": "q1", "depth": 0,
                          "utterance": "I'll give you eighty dollars"}})
    quote_id = int(quote.split("quote_id=")[1].split(",")[0])
    model.script = [
        AIMessage(content="", tool_calls=[{"name": "place_order", "id": "c2", "type": "tool_call",
                                           "args": {"shoe_id": 102, "customer_id": mine["CustomerID"],
                                                    "quote_id": quote_id}}]),
        AIMessage(content="Done, that pair is yours."),
    ]
    agreed = sse((await shop.post("/api/chat", json={"text": "okay, deal"})).text)
    meta = agreed[-1]["done"]
    assert meta["ordered"] and meta["order_id"], meta
    with db() as conn:
        haggled = dict(conn.execute("SELECT * FROM OrderDetails WHERE OrderID=?",
                                    (meta["order_id"],)).fetchone())
    assert haggled["Amount"] == 108 and haggled["ListPrice"] == 119.99, haggled
    paid_next = await shop.post(f"/orders/{meta['order_id']}/pay", data={"csrf": csrf})
    assert paid_next.headers["location"].endswith("?sent=1"), paid_next.headers
    with db() as conn:
        asked = conn.execute("SELECT AmountKes FROM Payment WHERE OrderID=?",
                             (meta["order_id"],)).fetchone()["AmountKes"]
    assert asked == mpesa.shillings(108) == 14040, asked
    print(f"7. haggling on the web: Mo quoted 120, took 108, and the prompt asked KSh {asked:,} - the "
          f"haggled price, not the shelf price")

    # ---------- 8. the chat is not a free model for strangers ----------
    model.script = [AIMessage(content="Still here.")] * 40
    tripped = False
    for _ in range(25):
        if (await shop.post("/api/chat", json={"text": "hello"})).status_code == 429:
            tripped = True
            break
    assert tripped, "the chat let one browser run 25 turns without protest"
    print("8. the chat is rate limited, so nobody burns the shop's free-tier tokens")

    # ---------- 9. knowing a customer's name and number is not being them ----------
    with db() as conn:
        victim = dict(conn.execute("SELECT CustomerID, CustomerName, Phone FROM CustomerInfo "
                                   "WHERE Source='web' ORDER BY CustomerID LIMIT 1").fetchone())
    thief = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shop",
                              follow_redirects=False)
    form = await thief.get("/profile")
    claimed = await thief.post("/profile", data={"name": victim["CustomerName"],
                                                 "phone": victim["Phone"],
                                                 "csrf": csrf_from(form.text), "next": "/"})
    assert claimed.status_code == 303, claimed.status_code
    theirs = await thief.get("/orders")
    assert "Order" not in theirs.text or "no orders" in theirs.text.lower(), theirs.text[:400]
    with db() as conn:
        rows = conn.execute("SELECT CustomerID FROM CustomerInfo WHERE Phone=? AND lower(CustomerName)"
                            "=lower(?)", (victim["Phone"], victim["CustomerName"])).fetchall()
    assert len(rows) == 2, "the second browser was handed the first one's customer record"
    # and a redirect cannot be aimed off the shop
    away = await thief.post("/profile", data={"name": victim["CustomerName"], "phone": victim["Phone"],
                                              "csrf": csrf_from(form.text), "next": "//evil.example"})
    assert away.headers["location"] == "/", away.headers
    await thief.aclose()
    print("9. a second browser typing the same name and number gets a profile of its own, sees none "
          "of the first one's orders, and cannot be redirected off the shop")

    await shop.aclose()
    await other.aclose()
    tools.wait_for_emails()
    assert _safety.real_outbox_snapshot() == REAL_OUTBOX_BEFORE, "the test wrote into the real outbox"
    print("10. the real outbox is untouched")
    print("\nALL WEB ASSERTIONS PASSED")


asyncio.run(main())
