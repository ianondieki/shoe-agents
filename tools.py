"""Agent tools: DB read/insert/update/delete, web search, email.

Return a string for a domain outcome ("out of stock", "no customer found") and let
infrastructure failures (network, SMTP) raise: agent_core records the exception as a
tool error before feeding it back to the model, so nothing fails silently.
"""
import json
import math
import os
import smtplib
import threading
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from db import db
from negotiation import ACCEPT, FINAL, HOLD, QUOTE_TTL_MINUTES, decide, spoken_offers
from tracing import TRACER, ctx


def _rows(rows):
    return [dict(r) for r in rows]


# ---------- READ ----------
@tool
def get_customer_info(customer_name: str) -> str:
    """Look up customers by full or partial name. Returns ID, email, city, preferred activity and shoe size."""
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM CustomerInfo WHERE CustomerName LIKE ?", (f"%{customer_name}%",)
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
                lower = [n for n in spoken_offers(utterance)
                         if 0.25 * shoe["Price"] <= n < quoted["OfferedPrice"]]
                if lower:
                    return (f"Not ordered: the customer just offered {lower[0]:.0f} dollars, not the "
                            f"{quoted['OfferedPrice']:.0f} you quoted. Put their number through "
                            f"evaluate_offer, tell them the answer, and order only once they agree.")
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
        if len(customers) != 1:
            # Ambiguous or missing: say so rather than guessing which person is calling.
            return json.dumps({"customer": None, "matches": len(customers), "shoes": [],
                               "note": "ask for the full name" if customers else "no such customer"})
        customer = customers[0]
        wanted = activity or customer.get("PreferredActivity") or ""
        shoes = _rows(conn.execute(
            "SELECT ShoeID, StyleDesc, ShoeColors, Price, InvCount FROM ShoeInventory "
            "WHERE BestFitActivity LIKE ? AND InvCount > 0 ORDER BY Price",
            (f"%{wanted}%",),
        ).fetchall())
    # The price as it is SAID: the whole dollar the haggling policy asks from. Handed 119.99, the
    # model rounded it to 119 on one call and 120 on the next, so the customer heard one number and
    # was negotiated from another. (place_order never charges above the true shelf price.)
    for s in shoes:
        s["Price"] = math.ceil(s["Price"] - 1e-9)
    return json.dumps({"customer": customer, "shoes": shoes})


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
    """Send, or dry-run to ./outbox when SMTP is not configured. Raises on a real SMTP failure."""
    user, pwd = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    if not (user and pwd):
        out = Path("outbox")
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
                return
            saved = (f" (list price {o['ListPrice']:.2f}, so you saved {o['ListPrice'] - o['Amount']:.2f})"
                     if o["ListPrice"] > o["Amount"] else "")
            body = (f"Hi {o['CustomerName']},\n\nYour order {o['OrderID']} is confirmed: one pair of "
                    f"the {o['StyleDesc']} for {o['Amount']:.2f}{saved}.\n\nThank you for shopping with us.\n"
                    f"Mo's shoe shop")
            result = _send_email(o["Email"], f"Order {o['OrderID']} confirmed", body)
            TRACER.event("email", order_id=order_id, ok=True, result=result)
        except Exception as e:  # the order stands either way; the failure is recorded, not lost
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


TOOLS = [
    get_customer_info, check_shoe_inventory, list_orders,
    place_order, cancel_order, delete_order,
    search_web, send_email,
]
# These need a human "y" before they run
SENSITIVE = {"place_order", "cancel_order", "delete_order", "send_email"}
