"""Agent tools: DB read/insert/update/delete, web search, email.

Return a string for a domain outcome ("out of stock", "no customer found") and let
infrastructure failures (network, SMTP) raise: agent_core records the exception as a
tool error before feeding it back to the model, so nothing fails silently.
"""
import json
import math
import os
import re
import smtplib
import sqlite3
import threading
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import mpesa
from db import db
from negotiation import ACCEPT, FINAL, HOLD, QUOTE_TTL_MINUTES, decide, said_yes, spoken_offers
from tracing import TRACER, ctx


def _rows(rows):
    return [dict(r) for r in rows]


# ---------- READ ----------
@tool
def get_customer_info(customer_name: str) -> str:
    """Look up customers by full or partial name. Returns ID, email, city, preferred activity and shoe size."""
    with db() as conn:
        # Named columns, not SELECT *: a partial name matches several people, and an address, an
        # email and a phone number are not things to hand a model that is talking to a stranger.
        rows = conn.execute(
            "SELECT CustomerID, CustomerName, City, PreferredActivity, ShoeSize FROM CustomerInfo "
            "WHERE CustomerName LIKE ?", (f"%{customer_name}%",)
        ).fetchall()
    return json.dumps(_rows(rows)) if rows else "No customer found."


@tool
def check_shoe_inventory(activity: str = "") -> str:
    """List shoes with price, colors and stock. Optionally filter by activity (e.g. Running, Hiking)."""
    with db() as conn:
        # Named columns, never SELECT *: FloorPrice lives in this table and must not reach a model.
        rows = conn.execute(
            "SELECT ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount "
            "FROM ShoeInventory WHERE BestFitActivity LIKE ?", (f"%{activity}%",)
        ).fetchall()
    return json.dumps(_rows(rows))


@tool
def list_orders(customer_id: int) -> str:
    """List all orders (placed and cancelled) for a customer ID, with what each one cost."""
    with db() as conn:
        # o.Amount, not s.Price: what the customer was charged, not what the shoe costs today.
        rows = conn.execute(
            """SELECT o.OrderID, o.OrderDate, o.Status, o.Amount, o.PaymentStatus,
                      s.ShoeID, s.StyleDesc
               FROM OrderDetails o JOIN ShoeInventory s ON s.ShoeID = o.ShoeID
               WHERE o.CustomerID = ? ORDER BY o.OrderID""",
            (customer_id,),
        ).fetchall()
    return json.dumps(_rows(rows)) if rows else "No orders for this customer."


# ---------- NEGOTIATE ----------
@tool
def evaluate_offer(shoe_id: int, customer_id: int, customer_offer: float, config: RunnableConfig) -> str:
    """Put a price the customer named through the shop's pricing rules.

    Use this for EVERY number a customer proposes. It returns the only price you are allowed to
    say out loud, and a quote_id to place the order against once they agree.
    """
    where = ctx(config)
    session = where["thread"] or "default"
    now = datetime.now()
    with db() as conn:
        shoe = conn.execute(
            "SELECT Price, FloorPrice FROM ShoeInventory WHERE ShoeID=?", (shoe_id,)
        ).fetchone()
        if not shoe:
            return f"Shoe {shoe_id} does not exist."
        if not conn.execute("SELECT 1 FROM CustomerInfo WHERE CustomerID=?", (customer_id,)).fetchone():
            return f"Customer {customer_id} does not exist."
        # Everything is decided from EARLIER turns only. One caller turn is one step of the haggle,
        # however many numbers it contains and however many times the model calls this tool - so
        # "would you take 80? 85? 90?" cannot walk the price to the floor in a single breath.
        this_turn = where["turn_id"]
        earlier = conn.execute(
            """SELECT OfferedPrice, Verdict, ListPrice FROM PriceQuote
               WHERE SessionID=? AND ShoeID=? AND CustomerID=? AND Status IN ('OPEN','SUPERSEDED')
                 AND CreatedTurnID IS NOT ?
               ORDER BY QuoteID DESC LIMIT 1""",
            (session, shoe_id, customer_id, this_turn),
        ).fetchone()
        # The round is how many turns the shop has actually conceded on. A quote at the shelf price
        # is not a concession, so a model that answers "how much is it?" by evaluating an offer the
        # customer never made does not use up the customer's first round.
        conceded = conn.execute(
            """SELECT COUNT(DISTINCT CreatedTurnID) FROM PriceQuote
               WHERE SessionID=? AND ShoeID=? AND CustomerID=? AND Verdict='COUNTER'
                 AND CreatedTurnID IS NOT ?""",
            (session, shoe_id, customer_id, this_turn),
        ).fetchone()[0]
        asking = earlier["OfferedPrice"] if earlier else shoe["Price"]
        offer = float(customer_offer)
        # A counter-offer must be one the CALLER made. In a live call the customer said only "that is
        # too much" and the model invented an offer of 100 on their behalf, so the shop conceded to
        # 110 unprompted - bargaining against itself. Agreeing to the shop's own price needs no
        # number from the caller; undercutting it does.
        utterance = (config or {}).get("configurable", {}).get("utterance")
        if (utterance is not None and offer < math.ceil(asking - 1e-9)
                and not any(abs(n - offer) < 0.5 for n in spoken_offers(utterance, money_only=False))):
            return (f"Not checked: the customer has not named {offer:.0f} dollars. Do not suggest a "
                    f"lower price yourself - ask what they had in mind, and put through only a number "
                    f"they actually say.")
        # Once the shop has agreed a discount or called its last price, that is where the haggle
        # ended: a later, lower number cannot reopen it. An "acceptance" at the full shelf price
        # agrees nothing, so it does not lock the price.
        locked = earlier is not None and (
            earlier["Verdict"] == FINAL
            or (earlier["Verdict"] == ACCEPT and earlier["OfferedPrice"] < earlier["ListPrice"]))
        if locked:
            price = int(earlier["OfferedPrice"])
            verdict = ACCEPT if offer >= price else FINAL
        else:
            verdict, price = decide(asking, shoe["FloorPrice"], offer, conceded + 1)
        round_n = conceded + 1

        conn.execute(
            "UPDATE PriceQuote SET Status='SUPERSEDED' "
            "WHERE SessionID=? AND ShoeID=? AND CustomerID=? AND Status='OPEN'",
            (session, shoe_id, customer_id),
        )
        cur = conn.execute(
            """INSERT INTO PriceQuote (SessionID, CustomerID, ShoeID, ListPrice, OfferedPrice,
                   CustomerOffer, Round, Verdict, Status, CreatedTurnID, CreatedAt, ExpiresAt)
               VALUES (?,?,?,?,?,?,?,?,'OPEN',?,?,?)""",
            (session, customer_id, shoe_id, shoe["Price"], price, float(customer_offer), round_n,
             verdict, where["turn_id"], now.isoformat(timespec="seconds"),
             (now + timedelta(minutes=QUOTE_TTL_MINUTES)).isoformat(timespec="seconds")),
        )
        quote_id = cur.lastrowid
    note = {
        FINAL: " This is your last price; do not go lower.",
        HOLD: " That offer is too low to consider: the price has not moved. Say so kindly.",
    }.get(verdict, "")
    return (f"{verdict} at {price} dollars (quote_id={quote_id}, round {round_n}).{note} "
            f"Say {price} dollars and no other number. If the customer then agrees, "
            f"call place_order with quote_id={quote_id}.")


# ---------- INSERT ----------
@tool
def place_order(shoe_id: int, customer_id: int, config: RunnableConfig, quote_id: int | None = None) -> str:
    """Place an order for one pair of shoes and reduce stock by one.

    Pass quote_id when a price was negotiated; the order is then charged at the quoted price.
    """
    where = ctx(config)
    # On a channel that knows who it is talking to - a signed-in browser, a WhatsApp number - the
    # model may only order for that person. Otherwise a typed "order for customer 3" puts a pair,
    # and its confirmation email, on a stranger's record.
    serving = ((config or {}).get("configurable", {}) or {}).get("customer_id")
    if serving and int(customer_id) != int(serving):
        return (f"Not ordered: you are serving customer_id={serving}, so you cannot order for "
                f"customer {customer_id}. Use customer_id={serving}.")
    now_iso = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        # Every read-only check happens before any write, because db() commits on return:
        # an early return after a mutation would persist a half-finished order.
        if not conn.execute("SELECT 1 FROM CustomerInfo WHERE CustomerID=?", (customer_id,)).fetchone():
            return f"Customer {customer_id} does not exist."
        shoe = conn.execute(
            "SELECT Price, InvCount FROM ShoeInventory WHERE ShoeID=?", (shoe_id,)
        ).fetchone()
        if not shoe or shoe["InvCount"] < 1:
            return f"Shoe {shoe_id} is out of stock or does not exist."

        amount = shoe["Price"]
        if quote_id is not None:
            # "The customer answered" is not "the customer agreed". If the words the model is
            # treating as a yes name a LOWER price, they are a counter-offer: refuse, and send the
            # model back to evaluate_offer. Only amounts in a money context count, and only ones a
            # quarter of the list price or more, so "size 8, two pairs" is never an offer.
            quoted = conn.execute(
                "SELECT OfferedPrice FROM PriceQuote WHERE QuoteID=?", (quote_id,)
            ).fetchone()
            utterance = (config or {}).get("configurable", {}).get("utterance") or ""
            if quoted:
                lower = [n for n in spoken_offers(utterance, bare=True)
                         if 0.25 * shoe["Price"] <= n < quoted["OfferedPrice"]]
                if lower:
                    return (f"Not ordered: the customer just offered {lower[0]:.0f} dollars, not the "
                            f"{quoted['OfferedPrice']:.0f} you quoted. Put their number through "
                            f"evaluate_offer, tell them the answer, and order only once they agree.")
                # ...and an answer that is not a yes is not consent either (see negotiation.said_yes).
                if utterance and not said_yes(utterance):
                    return (f"Not ordered: the customer has not clearly said yes - their words were "
                            f"\"{utterance[:80]}\". Ask them plainly whether to put it through at "
                            f"{quoted['OfferedPrice']:.0f} dollars, and order only when they say yes.")
            # Consent, enforced rather than trusted. CreatedTurnID IS NOT <this turn> means the
            # price must have been quoted on an EARLIER turn - so the customer heard it and
            # answered. Status='OPEN' makes a retried call a no-op instead of a second order.
            accepted = conn.execute(
                """UPDATE PriceQuote SET Status='ACCEPTED', AcceptedTurnID=?
                   WHERE QuoteID=? AND ShoeID=? AND CustomerID=? AND Status='OPEN'
                     AND ExpiresAt > ? AND CreatedTurnID IS NOT ?
                     AND OfferedPrice >= (SELECT FloorPrice FROM ShoeInventory WHERE ShoeID=?)""",
                (where["turn_id"], quote_id, shoe_id, customer_id, now_iso, where["turn_id"], shoe_id),
            )
            if accepted.rowcount == 0:
                return ("That quote cannot be ordered against: it is not open for this customer "
                        "and shoe, it has expired, or it was quoted on this same turn. Say the "
                        "price to the customer and order only after they agree.")
            # Prices are quoted in whole dollars, so a quote can sit a few cents above the shelf price
            # ("120" for a 119.99 pair). The customer is never charged more than the shelf price.
            amount = min(conn.execute(
                "SELECT OfferedPrice FROM PriceQuote WHERE QuoteID=?", (quote_id,)
            ).fetchone()["OfferedPrice"], shoe["Price"])

        # Only decrement if stock is available (the original code could go negative).
        if conn.execute(
            "UPDATE ShoeInventory SET InvCount = InvCount - 1 WHERE ShoeID=? AND InvCount > 0",
            (shoe_id,),
        ).rowcount == 0:
            raise RuntimeError(f"stock for shoe {shoe_id} vanished mid-order")  # rolls everything back

        # Snapshot both prices: Amount is what was charged, ListPrice what it would have cost.
        cur = conn.execute(
            """INSERT INTO OrderDetails (OrderDate, ShoeID, CustomerID, Amount, ListPrice, QuoteID)
               VALUES (?,?,?,?,?,?)""",
            (datetime.today().strftime("%Y-%m-%d"), shoe_id, customer_id, amount, shoe["Price"], quote_id),
        )
        order_id = cur.lastrowid
        if quote_id is not None:
            conn.execute("UPDATE PriceQuote SET Status='USED', OrderID=? WHERE QuoteID=?",
                         (order_id, quote_id))
    if quote_id is None:
        return f"Order {order_id} placed for {amount:.2f}."
    # A negotiated order confirms itself: written from the order row, to the address on file, off
    # the conversation's critical path. The model never chooses a recipient or writes the body, so
    # it cannot mail the wrong customer, promise a refund in writing, or confirm an order that does
    # not exist - and the caller is not left in silence while Gmail takes eight seconds.
    confirm_order_async(order_id)
    return (f"Order {order_id} placed for {amount:.2f}. A confirmation email is on its way to the "
            f"address on file; tell the customer, but do not read the address out.")


# ---------- UPDATE ----------
@tool
def cancel_order(order_id: int) -> str:
    """Cancel a placed order and return the pair to stock."""
    with db() as conn:
        row = conn.execute(
            "SELECT ShoeID FROM OrderDetails WHERE OrderID=? AND Status='PLACED'", (order_id,)
        ).fetchone()
        if not row:
            return f"Order {order_id} not found or already cancelled."
        conn.execute("UPDATE OrderDetails SET Status='CANCELLED' WHERE OrderID=?", (order_id,))
        conn.execute("UPDATE ShoeInventory SET InvCount = InvCount + 1 WHERE ShoeID=?", (row["ShoeID"],))
    return f"Order {order_id} cancelled and stock restored."


# ---------- DELETE ----------
@tool
def delete_order(order_id: int) -> str:
    """Permanently delete an order record. Only cancelled orders can be deleted."""
    with db() as conn:
        # A negotiated order is referenced by the quote it came from; unlink it first, or the
        # foreign key refuses the delete. The quote itself stays, as the record of the haggle.
        conn.execute(
            "UPDATE PriceQuote SET OrderID=NULL WHERE OrderID=? AND EXISTS "
            "(SELECT 1 FROM OrderDetails WHERE OrderID=? AND Status='CANCELLED')",
            (order_id, order_id),
        )
        cur = conn.execute(
            "DELETE FROM OrderDetails WHERE OrderID=? AND Status='CANCELLED'", (order_id,)
        )
    if cur.rowcount == 0:
        return f"Order {order_id} not deleted: it must exist and be cancelled first."
    return f"Order {order_id} deleted."


# ---------- COMPOSITE READ ----------
# Every tool call is its own model round trip (~600ms), so two 5ms lookups cost more than a
# second of inference between them. This collapses the usual opening pair into one rung.
# Not in TOOLS: the CLI agent keeps the granular tools; the voice agent binds this instead.
@tool
def customer_and_stock(customer_name: str, activity: str = "") -> str:
    """Look up a customer AND the shoes in stock that suit them, in ONE call. Always use this first.

    Returns {"customer": {...}, "shoes": [...]}. Shoes are filtered by `activity`, or by the
    customer's own preferred activity when `activity` is empty. Only in-stock shoes are returned.
    """
    with db() as conn:
        customers = _rows(conn.execute(
            "SELECT * FROM CustomerInfo WHERE CustomerName LIKE ?", (f"%{customer_name}%",)
        ).fetchall())
        # Stock does not depend on who is calling. Returning an empty shoe list for an unknown
        # caller made the model tell a live caller the trail runner was "out of stock" - it never was.
        customer = customers[0] if len(customers) == 1 else None
        in_stock = _rows(conn.execute(
            "SELECT ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount FROM ShoeInventory "
            "WHERE InvCount > 0 ORDER BY Price"
        ).fetchall())
    # Match what the caller asked for against the activity AND the name and colours: live, the model
    # searched "trail", which is in "Cushioned trail runner" but in no activity, and found nothing.
    wanted = activity or (customer or {}).get("PreferredActivity") or ""
    terms = [w.rstrip("s") for w in re.findall(r"[a-z]+", wanted.lower()) if len(w) > 2]
    shoes = [s for s in in_stock if any(
        t in f"{s['BestFitActivity']} {s['StyleDesc']} {s['ShoeColors']}".lower() for t in terms)]
    note = None
    if not shoes:
        shoes = in_stock  # nothing matched: offer everything that IS in stock, never an empty list
        if terms:
            note = f"Nothing matched {wanted!r}; these are all the shoes in stock."
    if customer is None:
        # Never read other customers' names to an unidentified caller.
        who = (f"No customer called {customer_name!r} is on file" if not customers
               else f"{len(customers)} customers match {customer_name!r}")
        note = (f"{who}. Say so, and ask them to spell their full name. You can tell them about the "
                f"shoes below - they ARE in stock. For an order you need a customer_id: if they are "
                f"nobody you can find, offer to put them on file with create_customer, using the "
                f"name and M-Pesa number they say themselves and only once they agree."
                + (f" {note}" if note else ""))
    result = {"customer": customer, "shoes": shoes}
    if note:
        result["note"] = note
    # The price as it is SAID: the whole dollar the haggling policy asks from. Handed 119.99, the
    # model rounded it to 119 on one call and 120 on the next, so the customer heard one number and
    # was negotiated from another. (place_order never charges above the true shelf price.)
    for s in shoes:
        s["Price"] = math.ceil(s["Price"] - 1e-9)
    return json.dumps(result)


# ---------- WEB SEARCH ----------
_DDGS = None


def _ddgs():
    """DDGS() loads proxy/impersonation config and costs ~300ms; build it once."""
    global _DDGS
    if _DDGS is None:
        from ddgs import DDGS

        _DDGS = DDGS()
    return _DDGS


@tool
def search_web(query: str) -> str:
    """Search the web (DuckDuckGo) for reviews, shoe advice or general info. Returns top results."""
    results = _ddgs().text(query, max_results=3)
    return json.dumps(
        [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")} for r in results]
    )


# ---------- EMAIL ----------
def _send_email(to: str, subject: str, body: str) -> str:
    """Send, or dry-run to the outbox folder when SMTP is not configured. Raises on SMTP failure.

    OUTBOX_DIR exists so tests can keep their dry-runs to themselves: the test suites blank the
    Gmail credentials (so they never send real mail) and, before this setting, every order they
    placed wrote a confirmation into the project's real ./outbox - 84 files the owner never asked for.
    """
    user, pwd = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    if not (user and pwd):
        out = Path(os.getenv("OUTBOX_DIR") or "outbox")
        out.mkdir(exist_ok=True)
        f = out / f"{datetime.now():%Y%m%d-%H%M%S-%f}.txt"
        f.write_text(f"To: {to}\nSubject: {subject}\n\n{body}", encoding="utf-8")
        return f"DRY RUN: email saved to {f}"

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
    return f"Email sent to {to}."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (e.g. an order confirmation). Dry-runs to ./outbox if SMTP is not configured."""
    return _send_email(to, subject, body)


_EMAIL_THREADS: list[threading.Thread] = []
# order id -> (ok, what happened): "Email sent to ...", "DRY RUN: email saved to <file>", or the error.
# The terminal call reads it to tell the caller what really became of their confirmation.
EMAIL_RESULTS: dict[int, tuple[bool, str]] = {}


def confirm_order_async(order_id: int) -> threading.Thread:
    """Email the confirmation for an order on a background thread, from the order row alone."""
    def run():
        try:
            with db() as conn:
                o = conn.execute(
                    """SELECT o.OrderID, o.Amount, o.ListPrice, c.CustomerName, c.Email, s.StyleDesc
                       FROM OrderDetails o JOIN CustomerInfo c ON c.CustomerID = o.CustomerID
                       JOIN ShoeInventory s ON s.ShoeID = o.ShoeID WHERE o.OrderID = ?""",
                    (order_id,),
                ).fetchone()
            if not o or not o["Email"]:
                EMAIL_RESULTS[order_id] = (False, "no email address on file")
                return
            saved = (f" (list price {o['ListPrice']:.2f}, so you saved {o['ListPrice'] - o['Amount']:.2f})"
                     if o["ListPrice"] > o["Amount"] else "")
            body = (f"Hi {o['CustomerName']},\n\nYour order {o['OrderID']} is confirmed: one pair of "
                    f"the {o['StyleDesc']} for {o['Amount']:.2f}{saved}.\n\nThank you for shopping with us.\n"
                    f"Mo's shoe shop")
            result = _send_email(o["Email"], f"Order {o['OrderID']} confirmed", body)
            EMAIL_RESULTS[order_id] = (True, result)
            TRACER.event("email", order_id=order_id, ok=True, result=result)
        except Exception as e:  # the order stands either way; the failure is recorded, not lost
            EMAIL_RESULTS[order_id] = (False, f"{type(e).__name__}: {e}")
            TRACER.event("email", order_id=order_id, ok=False, error=f"{type(e).__name__}: {e}")

    t = threading.Thread(target=run, name=f"confirm-order-{order_id}", daemon=True)
    _EMAIL_THREADS.append(t)
    t.start()
    return t


def wait_for_emails(timeout: float = 30) -> None:
    """Block until background confirmation emails finish. For tests and clean shutdown."""
    for t in list(_EMAIL_THREADS):
        t.join(timeout)
    _EMAIL_THREADS[:] = [t for t in _EMAIL_THREADS if t.is_alive()]


# ---------- NEW CUSTOMERS ----------
def add_customer(name: str, phone: str, email: str = "", city: str = "", channel: str = "web") -> dict:
    """Put a customer on file, or hand back the one already there. The web form and the agent's
    create_customer tool both come through here, so a profile is made one way only.

    Returns the customer row. Raises ValueError if the name or the number is not usable.
    """
    name = " ".join((name or "").split())
    if len(name) < 2 or not re.search(r"[A-Za-z]{2}", name):
        return {"error": "that is not a name"}
    try:
        number = mpesa.phone_number(phone)
    except ValueError as e:
        return {"error": str(e)}
    with db() as conn:
        # The same person coming back: same number AND same name. The number alone is not enough -
        # a household shares one, and the three seed customers share Daraja's test number.
        #
        # Being handed somebody else's record is the risk here, because that record carries their
        # orders. A web form proves nothing - anyone can type a name and a number they read off a
        # parcel - so the web never matches an existing customer and never leaves one to be matched.
        # WhatsApp and a call at least have a person saying it themselves.
        mine = None if channel == "web" else conn.execute(
            "SELECT * FROM CustomerInfo WHERE Phone = ? AND lower(CustomerName) = lower(?) "
            "AND Source <> 'web' ORDER BY CustomerID LIMIT 1", (number, name)).fetchone()
        if mine:
            return {"customer": dict(mine), "existing": True}
        cur = conn.execute(
            """INSERT INTO CustomerInfo (CustomerName, Email, Phone, City, Source, CreatedAt)
               VALUES (?,?,?,?,?,?)""",
            (name, (email or "").strip() or None, number, (city or "").strip() or None, channel,
             datetime.now().isoformat(timespec="seconds")))
        row = conn.execute("SELECT * FROM CustomerInfo WHERE CustomerID = ?", (cur.lastrowid,)).fetchone()
    TRACER.event("customer_created", customer_id=row["CustomerID"], source=channel)
    return {"customer": dict(row), "existing": False}


def update_customer(customer_id: int, name: str, phone: str, email: str = "") -> dict:
    """Correct the name or number on a record this browser already owns, rather than collecting a
    second profile for the same person every time they fill the form in again."""
    name = " ".join((name or "").split())
    if len(name) < 2 or not re.search(r"[A-Za-z]{2}", name):
        return {"error": "that is not a name"}
    try:
        number = mpesa.phone_number(phone)
    except ValueError as e:
        return {"error": str(e)}
    with db() as conn:
        conn.execute("""UPDATE CustomerInfo SET CustomerName=?, Phone=?, Email=COALESCE(?, Email)
                        WHERE CustomerID=?""",
                     (name, number, (email or "").strip() or None, customer_id))
        row = conn.execute("SELECT * FROM CustomerInfo WHERE CustomerID=?", (customer_id,)).fetchone()
    if not row:
        return {"error": "that profile is gone"}
    return {"customer": dict(row), "existing": True}


@tool
def create_customer(name: str, phone: str, config: RunnableConfig, email: str = "") -> str:
    """Put a new customer on file so they can order. Use when a look-up finds nobody.

    Only with a name and an M-Pesa number the customer gave you themselves, and only after you have
    said both back to them and they agreed.
    """
    asked = (config or {}).get("configurable", {}) or {}
    utterance = asked.get("utterance")
    # On WhatsApp the channel knows the number the person is messaging from. That beats anything the
    # model heard or read, so it wins and is never asked for.
    known_phone = asked.get("channel_phone") or ""
    # A customer the model invented is a customer nobody can deliver to: the words in front of it
    # must carry the name and, unless the channel already knows it, the number.
    if utterance and not known_phone:
        if re.sub(r"\D", "", phone or "")[-9:] not in re.sub(r"\D", "", utterance):
            return ("Not created: they have not said that number. Ask for their M-Pesa number, read "
                    "it back, and create them once they say yes.")
    if utterance and (name or "").split() and name.split()[0].lower() not in utterance.lower():
        return ("Not created: they have not said that name. Ask what name the order should be in and "
                "use exactly what they say.")
    if utterance and not said_yes(utterance):
        return ("Not created: they have not agreed yet. Say the name and the number back to them, and "
                "create them when they say yes.")
    made = add_customer(name, known_phone or phone, email=email, channel=asked.get("channel", "phone"))
    if "error" in made:
        return (f"Not created: {made['error']}. Ask them to say it again - the number is where the "
                f"M-Pesa prompt will go.")
    who, number = made["customer"], made["customer"]["Phone"]
    if made["existing"]:
        return (f"Already on file: {who['CustomerName']}, customer_id={who['CustomerID']}. Use that "
                f"id; do not make a second one.")
    return (f"Customer {who['CustomerID']} is on file for {who['CustomerName']}, paying from the "
            f"number ending {number[-4:]}. Use customer_id={who['CustomerID']} from here on.")


# ---------- PAYMENT ----------
_MPESA = None


def _mpesa():
    """One Daraja client for the process, made on first use so a shop with no keys still runs."""
    global _MPESA
    if _MPESA is None:
        _MPESA = mpesa.Mpesa()
    return _MPESA


def request_push(order_id: int, phone: str = "") -> dict:
    """Send the M-Pesa prompt for one order. The amount comes from the order row - never from a
    model, a browser or a caller. Returns {"sent": ...} or {"error": ...}."""
    with db() as conn:
        order = conn.execute(
            """SELECT o.OrderID, o.Amount, o.PaymentStatus, o.Status, c.Phone
               FROM OrderDetails o JOIN CustomerInfo c ON c.CustomerID = o.CustomerID
               WHERE o.OrderID = ?""", (order_id,)).fetchone()
        if not order:
            return {"error": f"order {order_id} does not exist"}
        if order["PaymentStatus"] == "PAID":
            return {"error": f"order {order_id} is already paid"}
        if order["Status"] == "CANCELLED":
            return {"error": f"order {order_id} was cancelled"}
        number = phone or order["Phone"] or ""
        if not number:
            return {"error": "no M-Pesa number on file for them"}
        amount_kes = mpesa.shillings(order["Amount"])
        # Somebody else's number can be typed into the web form, so cap how often any one phone can
        # be made to ring. Three prompts in ten minutes is more than a real customer needs.
        lately = conn.execute(
            "SELECT COUNT(*) FROM Payment WHERE Phone=? AND RequestedAt > ?",
            (number, (datetime.now() - timedelta(minutes=10)).isoformat(timespec="seconds"))
        ).fetchone()[0]
    if lately >= 3:
        return {"error": "that number has had several prompts in the last few minutes; give it a "
                         "moment before sending another"}
    # One prompt at a time per order, and the database decides it: a unique index allows one PENDING
    # row per order, so two taps a millisecond apart cannot both get past this and charge twice. The
    # row is written BEFORE Safaricom is asked, with a placeholder id, and filled in afterwards.
    ticket = f"reserved:{order_id}:{uuid4().hex[:10]}"
    try:
        with db() as conn:
            conn.execute(
                """INSERT INTO Payment (OrderID, Phone, AmountKes, CheckoutRequestID, Status,
                                        RequestedAt)
                   VALUES (?,?,?,?, 'PENDING', ?)""",
                (order_id, number, amount_kes, ticket,
                 datetime.now().isoformat(timespec="seconds")))
    except sqlite3.IntegrityError:
        return {"error": "a prompt for this order is already on their phone", "waiting": True}
    try:
        sent = _mpesa().push(number, amount_kes, reference=f"Order{order_id}", description="Shoes")
    except (mpesa.MpesaUnavailable, ValueError) as e:
        with db() as conn:      # nothing was sent, so the reservation must not block the next try
            conn.execute("DELETE FROM Payment WHERE CheckoutRequestID=?", (ticket,))
        TRACER.event("payment", order_id=order_id, ok=False, error=str(e))
        return {"error": str(e)}
    with db() as conn:
        conn.execute("UPDATE Payment SET CheckoutRequestID=?, MerchantRequestID=? "
                     "WHERE CheckoutRequestID=?",
                     (sent["checkout_id"], sent["merchant_id"], ticket))
        conn.execute("UPDATE OrderDetails SET PaymentStatus='PENDING', CheckoutRequestID=? "
                     "WHERE OrderID=?", (sent["checkout_id"], order_id))
    TRACER.event("payment", order_id=order_id, ok=True, amount_kes=amount_kes,
                 checkout_id=sent["checkout_id"])
    return {"sent": True, "amount_kes": amount_kes, "phone": number,
            "checkout_id": sent["checkout_id"]}


def settle(checkout_id: str) -> str:
    """Ask Daraja what became of one prompt and write it down. Idempotent, and the only way an order
    becomes PAID.

    A callback is never believed on its own: Daraja does not sign them, so anyone who learns the URL
    could claim a payment. We always ask Daraja ourselves, over the authenticated API, and the amount
    it reports must match what the order asked for.
    """
    with db() as conn:
        row = conn.execute(
            """SELECT p.PaymentID, p.OrderID, p.Status, p.AmountKes, o.Amount
               FROM Payment p JOIN OrderDetails o ON o.OrderID = p.OrderID
               WHERE p.CheckoutRequestID = ?""", (checkout_id,)).fetchone()
    if not row:
        return f"ignored: no prompt of ours has checkout id {checkout_id}"
    if row["Status"] != "PENDING":
        return f"already settled: payment {row['PaymentID']} is {row['Status']}"
    try:
        said = _mpesa().query(checkout_id)
    except mpesa.MpesaUnavailable as e:
        return f"could not ask M-Pesa: {e}"
    if not said["settled"]:
        return "still waiting on the phone"
    paid, reason, code = said["paid"], said["reason"], said.get("code")
    receipt = said.get("receipt") or ""
    # Daraja's query answer carries no amount, so the amount we trust is the one we asked for; the
    # callback's amount, when there is one, must agree with it.
    if paid and said.get("amount") and int(said["amount"]) != int(row["AmountKes"]):
        paid, reason = False, (f"the amount paid ({said['amount']}) is not the "
                               f"{row['AmountKes']} this order asked for")
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        changed = conn.execute(
            """UPDATE Payment SET Status=?, ResultCode=?, Receipt=?, Reason=?, SettledAt=?
               WHERE PaymentID=? AND Status='PENDING'""",
            ("PAID" if paid else "FAILED", code, receipt or None, reason, now, row["PaymentID"]))
        if changed.rowcount == 0:
            return f"already settled: payment {row['PaymentID']}"
        if paid:
            conn.execute("UPDATE OrderDetails SET PaymentStatus='PAID', MpesaReceipt=? WHERE OrderID=?",
                         (receipt or None, row["OrderID"]))
        else:
            # Not paid is not never asked: the order stands, unpaid, and can be prompted again.
            # Never from PAID, though - a later prompt that lapses must not reopen a paid order.
            conn.execute("UPDATE OrderDetails SET PaymentStatus='UNPAID' "
                         "WHERE OrderID=? AND PaymentStatus <> 'PAID'", (row["OrderID"],))
    TRACER.event("payment_settled", order_id=row["OrderID"], paid=paid, code=code, receipt=receipt,
                 reason=reason)
    return f"order {row['OrderID']}: {'paid' if paid else reason}"


def theirs(order_id: int, asked: dict) -> str:
    """Why this conversation may not touch that order, or "" when it may.

    A prompt makes somebody's phone ring and a status carries an M-Pesa receipt, so both are limited
    to the person the channel has identified: the number WhatsApp verified, or the customer a signed
    -in browser is. A conversation that is nobody in particular gets neither.
    """
    number, serving = asked.get("channel_phone") or "", asked.get("customer_id")
    if not number and not serving:
        return ("I have no verified number for this conversation, so I cannot send a prompt or read "
                "a payment from here. Ask them to pay on the shop's website or over WhatsApp.")
    with db() as conn:
        owner = conn.execute("""SELECT o.CustomerID, c.Phone FROM OrderDetails o
                                JOIN CustomerInfo c ON c.CustomerID = o.CustomerID
                                WHERE o.OrderID = ?""", (order_id,)).fetchone()
    if not owner:
        return f"Order {order_id} does not exist."
    if serving and int(owner["CustomerID"]) == int(serving):
        return ""
    if number and owner["Phone"] == number:
        return ""
    return (f"Order {order_id} is not theirs, so nothing about it can be sent or read here. Ask "
            f"which order they mean.")


@tool
def request_payment(order_id: int, config: RunnableConfig) -> str:
    """Send the M-Pesa prompt for an order to the customer's phone, for them to enter their PIN."""
    asked = (config or {}).get("configurable", {}) or {}
    refusal = theirs(order_id, asked)
    if refusal:
        return refusal
    out = request_push(order_id, phone=asked.get("channel_phone") or "")
    if out.get("waiting"):
        return ("A prompt is already on their phone for this order. Ask them to check it and enter "
                "their PIN, then use payment_status.")
    if "error" in out:
        return (f"The prompt did not go out: {out['error']}. Tell them plainly, and that the order is "
                f"held for them either way.")
    return (f"Sent: a prompt for {out['amount_kes']} shillings is on the phone ending "
            f"{out['phone'][-4:]}. Ask them to enter their M-Pesa PIN, then use payment_status.")


@tool
def payment_status(order_id: int, config: RunnableConfig) -> str:
    """What became of the M-Pesa prompt for an order: paid, still waiting, or why it failed."""
    refusal = theirs(order_id, (config or {}).get("configurable", {}) or {})
    if refusal:
        return refusal
    with db() as conn:
        order = conn.execute("SELECT PaymentStatus, MpesaReceipt FROM OrderDetails WHERE OrderID=?",
                             (order_id,)).fetchone()
        if not order:
            return f"Order {order_id} does not exist."
        if order["PaymentStatus"] == "PAID":
            return f"Order {order_id} is paid, M-Pesa receipt {order['MpesaReceipt']}. Thank them."
        last = conn.execute("SELECT Status, Reason, CheckoutRequestID FROM Payment WHERE OrderID=? "
                            "ORDER BY PaymentID DESC LIMIT 1", (order_id,)).fetchone()
    if not last:
        return f"No prompt has gone out for order {order_id} yet. Use request_payment."
    if last["Status"] != "PENDING":
        return (f"Order {order_id} is not paid: {last['Reason']}. Say so kindly and offer to send the "
                f"prompt again.")
    said = settle(last["CheckoutRequestID"])
    if "paid" in said and "not paid" not in said:
        return f"Order {order_id} is paid. Thank them."
    if "waiting" in said:
        return "The prompt is still on their phone; they have not finished with it. Give them a moment."
    return f"Order {order_id} is not paid: {said}. Offer to send the prompt again."


TOOLS = [
    get_customer_info, check_shoe_inventory, list_orders,
    place_order, cancel_order, delete_order,
    create_customer, request_payment, payment_status,
    search_web, send_email,
]
# These need a human "y" before they run
SENSITIVE = {"place_order", "cancel_order", "delete_order", "send_email", "create_customer",
             "request_payment"}
