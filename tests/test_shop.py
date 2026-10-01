"""Customers who sign themselves up, and orders that get paid - with a fake Daraja, so no money.

The guards are the point. A profile must come from what the customer actually said, an order's price
must come from the order row and nowhere else, and an order becomes PAID only when Safaricom's own
API says so - never because something POSTed our callback.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
DB = os.path.join(SCRATCH, "shop_test.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = os.path.join(SCRATCH, "shop_test_trace.jsonl")
os.environ.update(MPESA_ENV="sandbox", MPESA_CONSUMER_KEY="key", MPESA_CONSUMER_SECRET="secret",
                  MPESA_SHORTCODE="174379", MPESA_PASSKEY="passkey",
                  MPESA_CALLBACK_URL="https://example.trycloudflare.com/mpesa/callback",
                  KES_PER_USD="130")
sys.path.insert(0, ROOT)

import httpx  # noqa: E402

import mpesa  # noqa: E402
import tools  # noqa: E402
from db import db, init_db  # noqa: E402

init_db(reset=True)


class Daraja:
    """Safaricom's sandbox, scripted. `outcome` is what query() will say next."""

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
                                             "MerchantRequestID": f"29115-{self.pushes}"})
        if request.url.path.endswith("/query"):
            return httpx.Response(200, json=self.outcome)
        raise AssertionError(request.url.path)


daraja = Daraja()
tools._MPESA = mpesa.Mpesa(httpx.Client(transport=httpx.MockTransport(daraja.handler)))


def heard(said: str, **extra):
    """A turn of the agent's, as the voice server sets it up: the caller's own words reach the tools."""
    return {"configurable": {"thread_id": "test", "turn_id": "t1", "depth": 0, "utterance": said,
                             **extra}}


# ---------- 1. a profile only from what the customer actually said ----------
new = tools.create_customer.invoke({"name": "Grace Wanjiru", "phone": "0712345678"},
                                   heard("Hi, I'm Grace Wanjiru, my number is 0712345678, yes go ahead"))
assert new.startswith("Customer ") and "ending 5678" in new, new
customer_id = int(new.split("Customer ")[1].split()[0])

# the same person again: the record they already have, never a second one
again = tools.create_customer.invoke({"name": "Grace Wanjiru", "phone": "+254712345678"},
                                     heard("It's Grace Wanjiru again, 0712345678, yes please"))
assert f"customer_id={customer_id}" in again and "Already on file" in again, again
with db() as conn:
    assert conn.execute("SELECT COUNT(*) FROM CustomerInfo WHERE Phone='254712345678'"
                        ).fetchone()[0] == 1, "a second profile was created for the same person"

# a number nobody said, a name nobody said, and a yes nobody gave: all refused
for args, said, why in (
    ({"name": "Grace Wanjiru", "phone": "0799999999"}, "I'm Grace Wanjiru, 0712345678, yes", "number"),
    ({"name": "Peter Mwangi", "phone": "0712345678"}, "I'm Grace Wanjiru, 0712345678, yes", "name"),
    ({"name": "Mary Atieno", "phone": "0711222333"}, "Mary Atieno, 0711222333 - hmm, how much is it?",
     "agreed"),
):
    out = tools.create_customer.invoke(args, heard(said))
    assert out.startswith("Not created") and why in out, (why, out)
with db() as conn:
    assert conn.execute("SELECT COUNT(*) FROM CustomerInfo").fetchone()[0] == 4, "an invented profile"
print("1. profiles: made from the customer's own words and a plain yes; invented names, invented "
      "numbers and unanswered questions all refused")

# ---------- 2. WhatsApp knows the number, so the model is not asked to ----------
wa = tools.create_customer.invoke({"name": "Brian Otieno", "phone": "0700000000"},
                                  heard("Brian Otieno, yes", channel_phone="254733444555",
                                        channel="whatsapp"))
assert wa.startswith("Customer ") and "ending 4555" in wa, wa
with db() as conn:
    row = conn.execute("SELECT Phone, Source FROM CustomerInfo WHERE CustomerName='Brian Otieno'"
                       ).fetchone()
assert row["Phone"] == "254733444555" and row["Source"] == "whatsapp", dict(row)
print("2. WhatsApp: the number the person is messaging from wins over anything the model typed")

# ---------- 3. the new customer can buy, and the order keeps its own price ----------
quote = tools.evaluate_offer.invoke({"shoe_id": 102, "customer_id": customer_id, "customer_offer": 80},
                                    heard("I'll give you eighty dollars"))
quote_id = int(quote.split("quote_id=")[1].split(",")[0])
placed = tools.place_order.invoke({"shoe_id": 102, "customer_id": customer_id, "quote_id": quote_id},
                                  heard("Okay, deal", turn_id="t2"))
assert placed.startswith("Order "), placed
order_id = int(placed.split("Order ")[1].split()[0])
with db() as conn:
    order = dict(conn.execute("SELECT * FROM OrderDetails WHERE OrderID=?", (order_id,)).fetchone())
assert order["PaymentStatus"] == "UNPAID" and order["Amount"] == 108, order
print(f"3. the new customer haggled 119.99 down to {order['Amount']:.0f} and the order is UNPAID")

# ---------- 4. the prompt: the order's amount in shillings, to the customer's phone ----------
mine = {"customer_id": customer_id, "channel_phone": "254712345678"}
sent = tools.request_payment.invoke({"order_id": order_id}, heard("yes, send it", **mine))
assert "14040 shillings" in sent and "5678" in sent, sent
with db() as conn:
    pay = dict(conn.execute("SELECT * FROM Payment WHERE OrderID=?", (order_id,)).fetchone())
    assert pay["Status"] == "PENDING" and pay["AmountKes"] == 14040, pay
    assert conn.execute("SELECT PaymentStatus FROM OrderDetails WHERE OrderID=?",
                        (order_id,)).fetchone()["PaymentStatus"] == "PENDING"
twice = tools.request_payment.invoke({"order_id": order_id}, heard("send it again", **mine))
assert "already on their phone" in twice, twice
assert daraja.pushes == 1, "a second prompt went out while the first was still waiting"
print("4. the prompt: 108 dollars became 14,040 shillings on the phone ending 5678; no second prompt "
      "while that one is live")

# ---------- 5. nothing is paid until Safaricom's own API says so ----------
daraja.outcome = {"ResponseCode": "0", "errorMessage": "transaction is being processed"}
assert "still on their phone" in tools.payment_status.invoke(
    {"order_id": order_id}, heard("has it gone through?", **mine)).lower()

# a forged callback for a prompt we never sent changes nothing
assert "no prompt of ours" in tools.settle("ws_CO_forged")
# a callback for OUR prompt still proves nothing on its own: we ask Daraja, and it says cancelled
daraja.outcome = {"ResponseCode": "0", "ResultCode": "1032", "ResultDesc": "cancelled by user"}
said = tools.settle(pay["CheckoutRequestID"])
assert "cancelled" in said, said
with db() as conn:
    assert conn.execute("SELECT PaymentStatus FROM OrderDetails WHERE OrderID=?",
                        (order_id,)).fetchone()["PaymentStatus"] == "UNPAID"
    assert conn.execute("SELECT Status FROM Payment WHERE OrderID=?",
                        (order_id,)).fetchone()["Status"] == "FAILED"
print("5. a forged callback is ignored, and a real one that Daraja calls cancelled leaves the order "
      "UNPAID")

# ---------- 6. paid, once, however many times the news arrives ----------
again = tools.request_payment.invoke({"order_id": order_id}, heard("try again please", **mine))
assert "14040 shillings" in again, again
with db() as conn:
    live = conn.execute("SELECT CheckoutRequestID FROM Payment WHERE OrderID=? AND Status='PENDING'",
                        (order_id,)).fetchone()["CheckoutRequestID"]
daraja.outcome = {"ResponseCode": "0", "ResultCode": "0", "ResultDesc": "ok"}
assert "paid" in tools.settle(live)
assert "already settled" in tools.settle(live), "the same payment settled twice"
with db() as conn:
    order = dict(conn.execute("SELECT * FROM OrderDetails WHERE OrderID=?", (order_id,)).fetchone())
    paid_rows = conn.execute("SELECT Status FROM Payment WHERE OrderID=? ORDER BY PaymentID",
                             (order_id,)).fetchall()
assert order["PaymentStatus"] == "PAID", order
assert [r["Status"] for r in paid_rows] == ["FAILED", "PAID"], [dict(r) for r in paid_rows]
assert "is paid" in tools.payment_status.invoke({"order_id": order_id},
                                                heard("did that work?", **mine))
print("6. the second prompt was paid: the order is PAID once, both attempts kept, repeat news ignored")

# ---------- 7. an amount that does not match what the order asked for is not a payment ----------
assert "already paid" in tools.request_payment.invoke({"order_id": order_id},
                                                      heard("again", **mine))
with db() as conn:
    conn.execute("UPDATE OrderDetails SET PaymentStatus='UNPAID' WHERE OrderID=?", (order_id,))
    conn.execute("""INSERT INTO Payment (OrderID, Phone, AmountKes, CheckoutRequestID, Status,
                                         RequestedAt) VALUES (?,?,?,?, 'PENDING', ?)""",
                 (order_id, "254712345678", 14040, "ws_CO_short", "2026-10-01T10:00:00"))
short = mpesa.outcome(0, "ws_CO_short", receipt="SJK1", amount=100)   # 100 shillings, not 14,040
tools._MPESA.query = lambda cid: short                               # as if Daraja reported that
said = tools.settle("ws_CO_short")
assert "not the 14040" in said, said
with db() as conn:
    assert conn.execute("SELECT PaymentStatus FROM OrderDetails WHERE OrderID=?",
                        (order_id,)).fetchone()["PaymentStatus"] == "UNPAID"
print("7. 100 shillings against a 14,040 order is refused, not banked")

# ---------- 8. a prompt is only ever sent for the customer in front of you ----------
stranger = tools.create_customer.invoke(
    {"name": "Alice Njeri", "phone": "0722111222"},
    heard("I'm Alice Njeri, 0722111222, yes"))
alice = int(stranger.split("Customer ")[1].split()[0])
hers = tools.place_order.invoke({"shoe_id": 101, "customer_id": alice},
                                heard("yes, I'll take it", turn_id="t8", customer_id=alice,
                                      channel_phone="254722111222"))
her_order = int(hers.split("Order ")[1].split()[0])
before = daraja.pushes
for args, config, why in (
    ({"order_id": her_order}, heard("send the prompt", **mine), "someone else's order"),
    ({"order_id": her_order}, heard("send the prompt"), "nobody in particular"),
):
    out = tools.request_payment.invoke(args, config)
    assert "not theirs" in out or "no verified number" in out, (why, out)
assert daraja.pushes == before, "a prompt went out for an order that was not the caller's"
assert "not theirs" in tools.payment_status.invoke({"order_id": her_order},
                                                   heard("is it paid?", **mine))
# and the model cannot order for somebody it is not serving either
wrong = tools.place_order.invoke({"shoe_id": 101, "customer_id": alice},
                                 heard("yes please", turn_id="t9", **mine))
assert wrong.startswith("Not ordered"), wrong
print("8. an order belonging to another customer cannot be prompted, read or added to - not from "
      "this conversation, and not from an anonymous one")

# ---------- 9. a paid order stays paid, whatever happens to a later prompt ----------
with db() as conn:
    conn.execute("UPDATE OrderDetails SET PaymentStatus='PAID' WHERE OrderID=?", (order_id,))
    conn.execute("""INSERT INTO Payment (OrderID, Phone, AmountKes, CheckoutRequestID, Status,
                                         RequestedAt) VALUES (?,?,?,?, 'PENDING', ?)""",
                 (order_id, "254712345678", 14040, "ws_CO_late", "2026-10-01T10:05:00"))
tools._MPESA.query = lambda cid: mpesa.outcome(1032, "ws_CO_late")    # they cancelled this one
tools.settle("ws_CO_late")
with db() as conn:
    assert conn.execute("SELECT PaymentStatus FROM OrderDetails WHERE OrderID=?",
                        (order_id,)).fetchone()["PaymentStatus"] == "PAID", "a paid order reopened"
print("9. a later prompt that lapses cannot turn a paid order back into an unpaid one")

# ---------- 10. a profile typed into a web form speaks for nobody else ----------
web_row = tools.add_customer("Grace Wanjiru", "0712345678", channel="web")
assert not web_row["existing"] and web_row["customer"]["CustomerID"] != customer_id, web_row
from brain import customer_by_phone  # noqa: E402

assert customer_by_phone("0712345678")["customer_id"] == customer_id, "the web row answered for them"
spoken = tools.create_customer.invoke({"name": "Grace Wanjiru", "phone": "0712345678"},
                                      heard("Grace Wanjiru, 0712345678, yes"))
assert f"customer_id={customer_id}" in spoken, spoken
print("10. anyone can type a name and a number into a web form, so a web profile is never handed to "
      "another channel - and never hands over somebody's existing record")

print("\nALL SHOP ASSERTIONS PASSED")
