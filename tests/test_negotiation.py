"""The haggling policy and the consent guard, including a replay of the bug that motivated them.

In a live Phase 0 run the caller said "That's a bit steep. I'll give you eighty for it" and the
agent placed an order for 119.99 - the price they had just refused - because the approver trusted
the model. Everything below exists so that cannot happen again.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
os.makedirs(SCRATCH, exist_ok=True)
DB = os.path.join(SCRATCH, "negotiation.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = ""
sys.path.insert(0, ROOT)

from db import db, init_db  # noqa: E402
from negotiation import ACCEPT, COUNTER, FINAL, HOLD, decide  # noqa: E402
from tools import evaluate_offer, place_order  # noqa: E402
from voice_agent import voice_approve  # noqa: E402

REAL_OUTBOX_BEFORE = _safety.real_outbox_snapshot()
init_db(reset=True)
LIST, FLOOR = 119.99, 95.0  # shoe 102


def cfg(turn):
    return {"configurable": {"thread_id": "call-1", "turn_id": turn, "depth": 0}}


# ---------- 1. the policy alone: the floor is not reachable from any direction ----------
worst = None
for offer in range(0, 200, 1):
    for rnd in range(1, 8):
        verdict, price = decide(LIST, FLOOR, float(offer), rnd)
        assert verdict in (ACCEPT, COUNTER, FINAL, HOLD)
        if offer < FLOOR * 0.5:
            assert (verdict, price) == (HOLD, 120), f"a lowball of {offer} moved the price: {verdict} {price}"
        assert price >= FLOOR, f"floor breached: offer={offer} round={rnd} -> {price}"
        assert price <= LIST + 1, f"charged above list: {price}"
        assert price == int(price), f"not a whole dollar: {price}"
        worst = min(worst, price) if worst is not None else price
assert worst == 95, worst
print(f"policy: 1400 offer/round combinations, lowest price ever produced = {worst} (floor {FLOOR:.0f})")

# Each round's ask is the previous round's counter, so a real haggle chains. (Comparing rounds
# against a FIXED ask goes the other way, correctly: later rounds concede a smaller share.)
seq, ask = [], LIST
for r in (1, 2, 3):
    verdict, ask = decide(ask, FLOOR, 80.0, r)
    seq.append(ask)
assert seq[0] > seq[1] >= seq[2] >= FLOOR, seq
assert verdict == FINAL and seq[-1] == FLOOR, (verdict, seq)
# a shoe with no floor set is simply not negotiable
assert decide(LIST, 0, 10.0, 1) == (FINAL, 120)
print(f"concessions across rounds: {seq}; round 3 verdict is FINAL; floorless shoe is not negotiable")

# ---------- 2. THE BUG: ordering with no agreed price ----------
verdict = voice_approve("voice_shoe_agent", "place_order", {"shoe_id": 102, "customer_id": 1})
assert isinstance(verdict, str) and "no agreed price" in verdict.lower(), verdict
print("approver refuses place_order without a quote_id, with a recoverable reason")

# and the tool itself refuses a price it was simply handed
before = json.loads(evaluate_offer.invoke(
    {"shoe_id": 102, "customer_id": 1, "customer_offer": 80.0}, cfg("turn-1")).split("quote_id=")[1].split(",")[0])
assert before == 1

# ---------- 3. consent: a quote cannot be accepted on the turn it was created ----------
same_turn = place_order.invoke({"shoe_id": 102, "customer_id": 1, "quote_id": 1}, cfg("turn-1"))
assert "cannot be ordered against" in same_turn, same_turn
with db() as conn:
    assert conn.execute("SELECT COUNT(*) c FROM OrderDetails").fetchone()["c"] == 0
print("a quote created this turn cannot be ordered against - the customer has not replied yet")

# ---------- 4. grinding the same number does not move the price ----------
prices = []
for i, turn in enumerate(("turn-2", "turn-3", "turn-4"), start=2):
    out = evaluate_offer.invoke({"shoe_id": 102, "customer_id": 1, "customer_offer": 80.0}, cfg(turn))
    prices.append(int(out.split(" at ")[1].split(" dollars")[0]))
assert prices == sorted(prices, reverse=True), prices
assert min(prices) >= FLOOR, prices
assert "FINAL" in out, out
print(f"repeating the same 80 dollar offer five times: {[120] + prices} - never below {FLOOR:.0f}")

# ---------- 5. a price already conceded is never walked back up ----------
# We have already said 95 for shoe 102 in this session. A later, higher offer must not let the
# shop charge more than the number the customer was last told.
walked_up = evaluate_offer.invoke({"shoe_id": 102, "customer_id": 1, "customer_offer": 112.0}, cfg("turn-5"))
assert " at 95 dollars" in walked_up, walked_up
print("after conceding to 95, an offer of 112 is still quoted at 95 - the shop cannot walk its price back up")

# ---------- 6. the happy path, on a shoe with no history: quoted, then agreed next turn ----------
HIKE_LIST, HIKE_FLOOR = 149.99, 120.0  # shoe 103
quoted = evaluate_offer.invoke({"shoe_id": 103, "customer_id": 1, "customer_offer": 140.0}, cfg("turn-6"))
qid = int(quoted.split("quote_id=")[1].split(",")[0])
agreed_price = int(quoted.split(" at ")[1].split(" dollars")[0])
assert HIKE_FLOOR <= agreed_price < HIKE_LIST, agreed_price
ok = place_order.invoke({"shoe_id": 103, "customer_id": 1, "quote_id": qid}, cfg("turn-7"))
assert ok.startswith("Order 1 placed for"), ok
with db() as conn:
    o = conn.execute("SELECT * FROM OrderDetails WHERE OrderID=1").fetchone()
    q = conn.execute("SELECT Status, OrderID FROM PriceQuote WHERE QuoteID=?", (qid,)).fetchone()
    stock = conn.execute("SELECT InvCount FROM ShoeInventory WHERE ShoeID=103").fetchone()["InvCount"]
assert o["Amount"] == agreed_price, (o["Amount"], agreed_price)
assert o["ListPrice"] == HIKE_LIST and o["QuoteID"] == qid, dict(o)
assert q["Status"] == "USED" and q["OrderID"] == 1, dict(q)
assert stock == 2, stock
print(f"haggled 150 down to {o['Amount']:.0f}, agreed next turn, charged {o['Amount']:.0f} against list {o['ListPrice']}")

# ---------- 7. a retried order does not charge twice ----------
again = place_order.invoke({"shoe_id": 103, "customer_id": 1, "quote_id": qid}, cfg("turn-8"))
assert "cannot be ordered against" in again, again
with db() as conn:
    assert conn.execute("SELECT COUNT(*) c FROM OrderDetails").fetchone()["c"] == 1
    assert conn.execute("SELECT InvCount FROM ShoeInventory WHERE ShoeID=103").fetchone()["InvCount"] == 2
print("the same quote cannot be ordered twice - a retried voice turn is a no-op, not a second pair")

# ---------- 8. superseded quotes are dead, even if the model remembers the id ----------
evaluate_offer.invoke({"shoe_id": 101, "customer_id": 1, "customer_offer": 70.0}, cfg("turn-9"))
stale = int(evaluate_offer.invoke(
    {"shoe_id": 101, "customer_id": 1, "customer_offer": 70.0}, cfg("turn-10")).split("quote_id=")[1].split(",")[0]) - 1
assert "cannot be ordered against" in place_order.invoke(
    {"shoe_id": 101, "customer_id": 1, "quote_id": stale}, cfg("turn-11"))
print("an older quote from the same haggle is superseded and cannot be ordered against")

# ---------- 9. the voice agent cannot write email at all; orders confirm themselves ----------
# The review showed the model could mail any customer on file, with any body ("say it was 80 with
# free returns"). The fix is not a better check - it is that the capability is gone.
import glob  # noqa: E402

import voice_agent  # noqa: E402
from tools import wait_for_emails  # noqa: E402

assert "send_email" not in {t.name for t in voice_agent.TOOLS}, "the voice agent must not bind send_email"
wait_for_emails()
outbox = sorted(glob.glob(os.path.join(os.environ["OUTBOX_DIR"], "*.txt")), key=os.path.getmtime)
mail = open(outbox[-1], encoding="utf-8").read()
with db() as conn:
    jane = conn.execute("SELECT Email FROM CustomerInfo WHERE CustomerID=1").fetchone()["Email"]
assert mail.startswith(f"To: {jane}\n"), mail[:80]
assert "Order 1 confirmed" in mail and "145.00" in mail and "149.99" in mail, mail
print(f"9. no send_email on the voice path; order 1 emailed itself: {mail.splitlines()[1]!r} at 145.00")


def quote(shoe, offer, turn, customer=1, session="call-1"):
    out = evaluate_offer.invoke({"shoe_id": shoe, "customer_id": customer, "customer_offer": offer},
                                {"configurable": {"thread_id": session, "turn_id": turn, "depth": 0}})
    return int(out.split("quote_id=")[1].split(",")[0]), out.split(" at ")[0], int(out.split(" at ")[1].split(" dollars")[0])


def order(shoe, qid, turn, said="", customer=1, session="call-1"):
    return place_order.invoke({"shoe_id": shoe, "customer_id": customer, "quote_id": qid},
                              {"configurable": {"thread_id": session, "turn_id": turn, "depth": 0,
                                                "utterance": said}})


def fresh_db():
    """Reset the database - but first let any background confirmation email finish: on Windows
    a file another thread still has open cannot be deleted."""
    wait_for_emails()
    init_db(reset=True)


# ---------- 10. a reply that names a LOWER price is a refusal, not a yes (review: HIGH) ----------
fresh_db()
qid, _, price = quote(102, 120.0, "r-1")           # the model "checks" the shelf price
refused = order(102, qid, "r-2", said="That's a bit steep. I'll give you eighty for it.")
assert refused.startswith("Not ordered") and "80" in refused, refused
agreed = order(102, qid, "r-2", said="Okay, deal, I'll take it.")
assert agreed.startswith("Order 1 placed"), agreed
# quoted as "120 dollars", but a 119.99 pair is never charged more than 119.99
assert "119.99" in agreed, f"charged above the shelf price: {agreed}"
print(f"10. 'too steep, I'll give you eighty' is refused; 'okay, deal' orders: {agreed[:28]!r}")

# ---------- 11. one caller turn is one step, however many numbers it holds ----------
fresh_db()
steps = [quote(102, o, "m-1")[2] for o in (80.0, 85.0, 90.0)]   # "would you take 80? 85? 90?"
assert min(steps) > FLOOR, f"reached the floor in one breath: {steps}"
with db() as conn:
    rounds = {r["Round"] for r in conn.execute("SELECT Round FROM PriceQuote WHERE CreatedTurnID='m-1'")}
assert rounds == {1}, rounds
print(f"11. three offers in one turn stay in round 1: {steps}, never the floor of {FLOOR:.0f}")

# ---------- 12. a price question the model mis-files as an offer does not burn round 1 ----------
fresh_db()
quote(102, 120.0, "q-1")                              # "how much is it?" -> evaluated at the shelf price
_, verdict, price = quote(102, 80.0, "q-2")           # the real first offer
assert (verdict, price) == ("COUNTER", 108), (verdict, price)
print(f"12. after a mis-filed price question, the first real offer still gets round 1: {price}")

# ---------- 13. an agreed discount is where the haggle ends; it cannot be walked down ----------
fresh_db()
_, verdict, agreed_at = quote(102, 117.0, "w-1")      # within 3% of 120: accepted
assert verdict == "ACCEPT" and agreed_at == 117, (verdict, agreed_at)
walked = [quote(102, o, t)[1:] for o, t in ((110.0, "w-2"), (100.0, "w-3"))]
assert all(p == 117 for _, p in walked), walked
print(f"13. after agreeing 117, offers of 110 and 100 get {walked} - the price does not come back down")

# ---------- 14. absurd offers do not buy concessions, so they cannot find the floor ----------
fresh_db()
held = [quote(102, 1.0, f"h-{i}")[1:] for i in range(1, 5)]
assert all(v == "HOLD" and p == 120 for v, p in held), held
honest = [quote(102, 96.0, f"o-{i}", session="call-2")[1:] for i in range(1, 4)]
assert honest[-1][1] <= 105, honest
print(f"14. one dollar, four times: {held[-1]}; an honest 96 three times: {honest[-1]}")

# ---------- 15. every predicate in the consent UPDATE earns its place (review: mutation) ----------
fresh_db()
qa = quote(103, 140.0, "p-1")[0]
assert "cannot be ordered" in order(101, qa, "p-2"), "wrong shoe accepted"
assert "cannot be ordered" in order(103, qa, "p-2", customer=2), "wrong customer accepted"
with db() as conn:
    conn.execute("UPDATE PriceQuote SET ExpiresAt='2000-01-01T00:00:00' WHERE QuoteID=?", (qa,))
assert "cannot be ordered" in order(103, qa, "p-2"), "expired quote accepted"
qb = quote(103, 140.0, "p-3")[0]
with db() as conn:
    conn.execute("UPDATE ShoeInventory SET FloorPrice=148 WHERE ShoeID=103")  # owner raises the floor
assert "cannot be ordered" in order(103, qb, "p-4"), "quote below a raised floor accepted"
print("15. wrong shoe, wrong customer, expired, and below a raised floor: each refused on its own")

# ---------- 16. a negotiated order can still be cancelled and deleted (review: FK regression) ----------
from tools import cancel_order, check_shoe_inventory, delete_order  # noqa: E402

fresh_db()
qc = quote(103, 140.0, "d-1")[0]
assert order(103, qc, "d-2").startswith("Order 1 placed")
assert "cancelled" in cancel_order.invoke({"order_id": 1})
assert delete_order.invoke({"order_id": 1}) == "Order 1 deleted."
print("16. negotiated order: placed, cancelled and deleted without a foreign-key error")

# ---------- 17. the floor never reaches a model, on either path (review: CLI leak) ----------
assert "FloorPrice" not in check_shoe_inventory.invoke({"activity": ""})
from tools import customer_and_stock  # noqa: E402

assert "FloorPrice" not in customer_and_stock.invoke({"customer_name": "Jane Doe", "activity": ""})
print("17. FloorPrice is absent from every read tool's output")

# ---------- 18. an owner's 'not negotiable' survives a restart (review: backfill) ----------
with db() as conn:
    conn.execute("UPDATE ShoeInventory SET FloorPrice=0 WHERE ShoeID=102")
init_db()
with db() as conn:
    assert conn.execute("SELECT FloorPrice FROM ShoeInventory WHERE ShoeID=102").fetchone()[0] == 0
assert quote(102, 80.0, "n-1")[1:] == ("FINAL", 120)
print("18. FloorPrice=0 is still 0 after init_db, and the shoe is FINAL at its shelf price")

# ---------- 19. the model cannot make an offer on the customer's behalf ----------
# Live: the caller said only "that is too much", the model evaluated an invented 100, and the
# shop conceded to 110 unprompted.
fresh_db()


def said(offer, utterance, turn):
    return evaluate_offer.invoke({"shoe_id": 102, "customer_id": 1, "customer_offer": offer},
                                 {"configurable": {"thread_id": "call-9", "turn_id": turn, "depth": 0,
                                                   "utterance": utterance}})


invented = said(100.0, "That is too much.", "i-1")
assert invented.startswith("Not checked"), invented
with db() as conn:
    assert conn.execute("SELECT COUNT(*) FROM PriceQuote").fetchone()[0] == 0, "a refused offer wrote a quote"
assert said(80.0, "Eighty.", "i-2").startswith("COUNTER"), "a bare number after 'what price?' was refused"
assert said(85.0, "Would you do eighty five?", "i-3").startswith("COUNTER")
full = said(120.0, "Okay, I'll take it at the shelf price.", "i-4")
assert full.startswith("ACCEPT") or full.startswith("FINAL"), f"agreeing to the shop's price was refused: {full}"
print("19. 'too much' cannot become an invented offer of 100; 'Eighty.' and agreeing to the shop's price both work")

# ---------- 20. the first real voice call: an unknown caller heard "out of stock" for a shoe in stock ----------
fresh_db()
from tools import customer_and_stock as cas  # noqa: E402

unknown = json.loads(cas.invoke({"customer_name": "John Doe", "activity": ""}))
assert unknown["customer"] is None and unknown["shoes"], f"unknown caller got an empty stock list: {unknown}"
note = unknown["note"].lower()
assert "is on file" in note and "spell" in note, unknown["note"]
# and, since the shop now signs people up, the way to turn a stranger into a customer
assert "create_customer" in note, unknown["note"]
assert "Jane" not in unknown["note"] and "Kamau" not in unknown["note"], "leaked other customers' names"
print(f"20a. unknown caller 'John Doe': told to ask them to spell it, and still sees {len(unknown['shoes'])} shoes in stock")

trail = json.loads(cas.invoke({"customer_name": "Jane Doe", "activity": "trail"}))
assert [s["ShoeID"] for s in trail["shoes"]] == [102], f"'trail' should find the trail runner: {trail['shoes']}"
boots = json.loads(cas.invoke({"customer_name": "Jane Doe", "activity": "boots"}))
assert [s["ShoeID"] for s in boots["shoes"]] == [103], f"'boots' should find the mid boot: {boots['shoes']}"
nothing = json.loads(cas.invoke({"customer_name": "Jane Doe", "activity": "tennis"}))
assert len(nothing["shoes"]) == 4 and "Nothing matched" in nothing["note"], nothing
assert all(isinstance(s["Price"], int) for s in nothing["shoes"])
print("20b. 'trail' finds the trail runner by name, 'boots' finds the mid boot, 'tennis' offers all 4 in stock")

# ---------- 22. a terminal test call: speech-to-text turned a refusal into "Deep." and Mo ordered on it ----------
from negotiation import said_yes  # noqa: E402

fresh_db()
qid, _, price = quote(102, 120.0, "d-1")           # "How much is the trail runner?" - the model "checks" it
refused = order(102, qid, "d-2", said="Deep.")
assert refused.startswith("Not ordered") and "clearly said yes" in refused, refused
for no in ("No, that's fine.", "Okay, that's too much.", "Let me think about it.", "What colours has it got?"):
    assert order(102, qid, "d-2", said=no).startswith("Not ordered"), no
with db() as conn:
    assert conn.execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0] == 0, "an order went through"
agreed = order(102, qid, "d-3", said="Yes please, go ahead.")
assert agreed.startswith("Order"), agreed
yes = ["Okay, deal.", "Sure, go ahead.", "Alright, I'll take it.", "Why not!", "Why not?", "Put it through.",
       "108 it is.", "Deal, email me the confirmation.", "Okay, deal. Just go with the initial one.", "Hmm, okay.",
       "Can you put it through?", "Yes please, size 9."]
assert all(said_yes(s) for s in yes), [s for s in yes if not said_yes(s)]
print(f"22. 'Deep.' (a misheard refusal) and 4 other non-answers ordered nothing; 'Yes please, go ahead.' did; "
      f"{len(yes)} ways of saying yes all count")

# ---------- 23. review: a yes-word is not a yes when it opens a question or hides a lower number ----------
fresh_db()
qid, _, price = quote(102, 120.0, "q-1")
assert price == 120, price
for words in ("Okay, ninety.", "Why not ninety?",                         # a bare counter-offer
              "Okay, so what sizes do you have?", "Great, and what colours?",   # a question
              "Is that a good price?", "Alright, what's the lowest you can go?",
              "Yes, I'm here.",                                          # the answer to "still there?"
              "Can you put it through for less?"):
    out = order(102, qid, "q-2", said=words)
    assert out.startswith("Not ordered"), f"{words!r} placed an order: {out}"
with db() as conn:
    assert conn.execute("SELECT COUNT(*) FROM OrderDetails").fetchone()[0] == 0, "an order went through"
assert order(102, qid, "q-3", said="Yes please, size 9. Put it through.").startswith("Order"), "a size is not an offer"
print("23. 'Okay, ninety.', 'Why not ninety?', questions opening with a yes-word and 'Yes, I'm here.' order "
      "nothing; 'Yes please, size 9. Put it through.' does")

wait_for_emails()
assert _safety.real_outbox_snapshot() == REAL_OUTBOX_BEFORE, "a test wrote into the project's real outbox"
print("21. the project's real outbox is exactly as it was before the suite ran")

print("\nALL NEGOTIATION ASSERTIONS PASSED")
