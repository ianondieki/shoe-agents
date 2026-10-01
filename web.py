"""The shop as a web page: browse, haggle with Mo, order, pay by M-Pesa.

Everything a buyer does here goes through the same tools the phone agent uses - place_order,
add_customer, request_push - so there is one way to make an order and one place that decides a
price. The browser never sends an amount; it sends a shoe id and a click.

Identity without a password: a signed cookie carries the session, and a profile is a name and the
M-Pesa number the buyer pays with. The PIN prompt landing on that handset is the proof, and it is
free - an SMS code is not.
"""
import asyncio
import hmac
import json
import os
import re
import time
from urllib.parse import quote
import uuid
from hashlib import sha256

from fastapi import APIRouter, Form, Request
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.templating import Jinja2Templates

import mpesa
import tools
from brain import Brain
from db import db

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = Jinja2Templates(directory=os.path.join(ROOT, "templates"))
COOKIE = "mo_session"
SECRET = (os.getenv("WEB_SECRET") or "").encode() or os.urandom(32)
if not os.getenv("WEB_SECRET"):
    # A secret nobody set is a secret nobody knows - including us, next restart, when every
    # browser is quietly signed out. Fine locally, which is why this warns rather than refuses.
    print("  WEB_SECRET is not set: sessions last only until this process stops.")


# ---------- who is this, and is it really them? ----------
def sign(value: str) -> str:
    return hmac.new(SECRET, value.encode(), sha256).hexdigest()[:32]


def session_of(request: Request) -> str:
    """The browser's session id, or a fresh one. Signed, so it cannot be swapped for someone else's."""
    raw = request.cookies.get(COOKIE) or ""
    sid, _, mac = raw.partition(".")
    if sid and mac and hmac.compare_digest(mac, sign(sid)):
        return sid
    return uuid.uuid4().hex[:16]


def set_session(response: Response, sid: str) -> Response:
    response.set_cookie(COOKIE, f"{sid}.{sign(sid)}", max_age=60 * 60 * 24 * 30, httponly=True,
                        samesite="lax")
    return response


def csrf_for(sid: str) -> str:
    return sign("csrf:" + sid)


def checked(sid: str, token: str) -> bool:
    return bool(token) and hmac.compare_digest(token, csrf_for(sid))


def whoami(sid: str) -> dict:
    """The customer this browser has said it is. Kept in the database, not in the cookie, so a
    cookie on its own can never name someone else."""
    with db() as conn:
        row = conn.execute(
            """SELECT c.CustomerID, c.CustomerName, c.Phone FROM CustomerInfo c
               JOIN WebSession w ON w.CustomerID = c.CustomerID WHERE w.SID = ?""", (sid,)).fetchone()
    return ({"customer_id": row["CustomerID"], "name": row["CustomerName"], "phone": row["Phone"]}
            if row else {})


def remember(sid: str, customer_id: int):
    with db() as conn:
        conn.execute("""INSERT INTO WebSession (SID, CustomerID, SeenAt) VALUES (?,?,?)
                        ON CONFLICT(SID) DO UPDATE SET CustomerID=excluded.CustomerID,
                        SeenAt=excluded.SeenAt""",
                     (sid, customer_id, __import__("datetime").datetime.now().isoformat(
                         timespec="seconds")))


def here(path: str) -> str:
    """A redirect target that cannot leave the shop: a browser reads "//evil.example" and
    "/\\evil.example" as somewhere else entirely, however much they look like paths."""
    return path if path.startswith("/") and not path.startswith(("//", "/\\")) else "/"


def _settled(checkout_id: str):
    """settle() in a thread, with its errors said out loud: Daraja already has its 200, so a
    failure here is the only sign that a payment was never written down."""
    try:
        print(f"  M-Pesa: {tools.settle(checkout_id)}")
    except Exception as e:
        print(f"  M-Pesa callback for {checkout_id} failed: {type(e).__name__}: {e}")


# Tasks the loop would otherwise hold only weakly, and so could collect mid-await.
_ASKING: set = set()


# ---------- a small amount of shouting is still shouting ----------
_SEEN: dict[str, list[float]] = {}


def too_fast(request: Request, key: str, times: int = 20, per_s: float = 60.0) -> bool:
    """A plain token bucket per browser. Every turn costs free-tier tokens; nobody gets to burn them."""
    # The socket, not a header: cf-connecting-ip is written by whoever is calling, so a bucket
    # keyed on it is one made-up value away from unlimited.
    who = request.client.host if request.client else "?"
    now, hits = time.time(), _SEEN.setdefault(f"{key}:{who}", [])
    hits[:] = [t for t in hits if now - t < per_s]
    hits.append(now)
    if len(_SEEN) > 5000:        # one key per caller, so forget the ones that went quiet
        for stale in [k for k, v in _SEEN.items() if not v or now - v[-1] > per_s * 10]:
            _SEEN.pop(stale, None)
    return len(hits) > times


def shoes_in_stock() -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT ShoeID, StyleDesc, ShoeColors, BestFitActivity, Price, InvCount FROM ShoeInventory "
            "WHERE InvCount > 0 ORDER BY Price")]


def order_of(customer_id: int, order_id: int) -> dict | None:
    """One order, if it is theirs. Asked for by id rather than filtered out of the whole list."""
    for row in orders_for(customer_id, order_id):
        return row
    return None


def orders_for(customer_id: int, order_id: int | None = None) -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT o.OrderID, o.OrderDate, o.Amount, o.ListPrice, o.Status, o.PaymentStatus,
                      o.MpesaReceipt, s.StyleDesc
               FROM OrderDetails o JOIN ShoeInventory s ON s.ShoeID = o.ShoeID
               WHERE o.CustomerID = ? AND (? IS NULL OR o.OrderID = ?)
               ORDER BY o.OrderID DESC""", (customer_id, order_id, order_id))]


def router(brain: Brain) -> APIRouter:
    api = APIRouter()

    def page(request: Request, template: str, **context) -> Response:
        # Not "name": the profile page passes a customer's name through here as context.
        sid = session_of(request)
        html = TEMPLATES.TemplateResponse(request, template, {
            "me": whoami(sid), "csrf": csrf_for(sid), "kes": mpesa.shillings, **context})
        # Every page carries this browser's CSRF token and its own name on it. A cached copy means a
        # token from an older session, which looks exactly like an attack and bounces the form.
        html.headers["Cache-Control"] = "no-store"
        return set_session(html, sid)

    # ---------- the shop ----------
    @api.get("/", response_class=HTMLResponse)
    async def shop(request: Request, err: str = ""):
        return page(request, "shop.html", shoes=shoes_in_stock(), err=err)

    @api.post("/buy")
    async def buy(request: Request, shoe_id: int = Form(...), csrf: str = Form("")):
        sid = session_of(request)
        me = whoami(sid)
        if not checked(sid, csrf):
            return set_session(RedirectResponse("/?err=Try that again.", 303), sid)
        if not me:
            return set_session(RedirectResponse(f"/profile?next=/buy&shoe_id={shoe_id}", 303), sid)
        # The shelf price, read here and not sent by the browser. A lower price only ever comes from
        # a quote Mo made in the chat, and place_order checks that for itself.
        said = await asyncio.to_thread(tools.place_order.invoke,
                                       {"shoe_id": shoe_id, "customer_id": me["customer_id"]},
                                       {"configurable": {"thread_id": f"web:{sid}",
                                                         "turn_id": f"buy-{uuid.uuid4().hex[:6]}",
                                                         "depth": 0}})
        made = re.match(r"Order (\d+) ", said)
        if not made:
            return set_session(RedirectResponse(f"/?err={quote(said[:200])}", 303), sid)
        order_id = int(made.group(1))
        return set_session(RedirectResponse(f"/orders/{order_id}", 303), sid)

    # ---------- who you are ----------
    @api.get("/profile", response_class=HTMLResponse)
    async def profile_form(request: Request, next: str = "/", shoe_id: int = 0):
        return page(request, "profile.html", next=next, shoe_id=shoe_id, problem="", name="",
                    phone="", email="")

    @api.post("/profile")
    async def profile_save(request: Request, name: str = Form(""), phone: str = Form(""),
                           email: str = Form(""), csrf: str = Form(""), next: str = Form("/"),
                           shoe_id: int = Form(0)):
        sid = session_of(request)
        if not checked(sid, csrf):
            return set_session(RedirectResponse("/profile", 303), sid)
        mine = whoami(sid)
        if mine:
            made = await asyncio.to_thread(tools.update_customer, mine["customer_id"], name,
                                           phone, email)
        else:
            made = await asyncio.to_thread(tools.add_customer, name, phone, email, "", "web")
        if "error" in made:
            return set_session(TEMPLATES.TemplateResponse(request, "profile.html", {
                "me": {}, "csrf": csrf_for(sid), "next": next, "shoe_id": shoe_id,
                "problem": made["error"], "name": name, "phone": phone, "email": email,
                "kes": mpesa.shillings}), sid)
        remember(sid, made["customer"]["CustomerID"])
        if next == "/buy" and shoe_id:
            return set_session(page(request, "buying.html", shoe_id=shoe_id), sid)
        return set_session(RedirectResponse(here(next), 303), sid)

    # ---------- orders and money ----------
    @api.get("/orders", response_class=HTMLResponse)
    async def order_list(request: Request):
        me = whoami(session_of(request))
        return page(request, "orders.html", orders=orders_for(me["customer_id"]) if me else [])

    @api.get("/orders/{order_id}", response_class=HTMLResponse)
    async def order_page(request: Request, order_id: int, sent: int = 0, problem: str = ""):
        me = whoami(session_of(request))
        mine = order_of(me["customer_id"], order_id) if me else None
        return page(request, "order.html", order=mine, order_id=order_id,
                    sent=sent, problem=problem)

    @api.post("/orders/{order_id}/pay")
    async def pay(request: Request, order_id: int, csrf: str = Form("")):
        sid = session_of(request)
        me = whoami(sid)
        if not checked(sid, csrf):
            return set_session(RedirectResponse(f"/orders/{order_id}", 303), sid)
        # Only the browser the order belongs to can pay it, and an unknown browser owns nothing.
        if not me or not order_of(me["customer_id"], order_id):
            return set_session(RedirectResponse("/orders", 303), sid)
        out = await asyncio.to_thread(tools.request_push, order_id, me["phone"])
        where = (f"/orders/{order_id}?sent=1" if out.get("sent") or out.get("waiting")
                 else f"/orders/{order_id}?problem="
                      f"{quote(str(out.get('error', 'the prompt did not go out'))[:200])}")
        return set_session(RedirectResponse(where, 303), sid)

    @api.get("/api/orders/{order_id}/status")
    async def order_status(request: Request, order_id: int):
        """The order page watches this while the prompt is on the phone."""
        me = whoami(session_of(request))
        order = order_of(me["customer_id"], order_id) if me else None
        if not order:
            return JSONResponse({"error": "no such order"}, status_code=404)
        if order["PaymentStatus"] == "PENDING":
            with db() as conn:
                waiting = conn.execute("SELECT CheckoutRequestID FROM Payment WHERE OrderID=? AND "
                                       "Status='PENDING' ORDER BY PaymentID DESC LIMIT 1",
                                       (order_id,)).fetchone()
            if waiting:
                # Ask Safaricom rather than wait for a callback that may never arrive.
                await asyncio.to_thread(tools.settle, waiting["CheckoutRequestID"])
            order = order_of(me["customer_id"], order_id) or order
        return JSONResponse({"status": order["PaymentStatus"], "receipt": order["MpesaReceipt"] or "",
                             "amount": order["Amount"]})

    # ---------- talking to Mo ----------
    @api.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return page(request, "chat.html", voice=False)

    @api.get("/call", response_class=HTMLResponse)
    async def call_page(request: Request):
        return page(request, "chat.html", voice=True)

    @api.post("/api/chat")
    async def chat(request: Request):
        sid = session_of(request)
        if too_fast(request, "chat"):
            return JSONResponse({"error": "Too many messages at once. Give it a moment."},
                                status_code=429)
        try:
            asked = await request.json()
            said = str(asked.get("text") or "").strip()[:500]
        except Exception:
            return JSONResponse({"error": "I could not read that."}, status_code=400)
        me = whoami(sid)

        async def stream():
            async for kind, payload in brain.turn(f"web:{sid}", said, known=me or None, channel="web"):
                # "done" carries the turn id and whatever went wrong inside. The browser needs
                # neither, and an exception's text can name internals.
                if kind == "done":
                    payload = {"ordered": bool(payload.get("ordered")),
                               "order_id": payload.get("order_id")}
                yield f"data: {json.dumps({kind: payload})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/api/voice")
    async def voice(request: Request):
        """A recording from the browser: what they said, and what Mo says back."""
        import speech

        sid = session_of(request)
        if too_fast(request, "voice", times=10):
            return JSONResponse({"error": "Too many recordings at once."}, status_code=429)
        form = await request.form()
        clip = form.get("audio")
        if clip is None or not hasattr(clip, "read"):
            return JSONResponse({"error": "no audio"}, status_code=400)
        if int(request.headers.get("content-length") or 0) > 4_000_000:
            return JSONResponse({"error": "That recording is too long."}, status_code=413)
        data = await clip.read()
        ears = speech.Transcriber(vocabulary=speech.shop_vocabulary())
        try:
            heard = await ears.spoken(data, getattr(clip, "filename", "clip.webm"),
                                      getattr(clip, "content_type", "audio/webm"))
        except speech.SpeechUnavailable as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        finally:
            await ears.aclose()
        if not heard:
            return JSONResponse({"heard": "", "say": [], "error": "I didn't catch that."})
        me, lines, meta = whoami(sid), [], {}
        async for kind, payload in brain.turn(f"web:{sid}", heard, known=me or None, channel="web"):
            lines.append(payload) if kind == "say" else meta.update(payload)
        return JSONResponse({"heard": heard, "say": lines,
                             "done": {"ordered": bool(meta.get("ordered")),
                                      "order_id": meta.get("order_id")}})

    @api.get("/api/tts")
    async def tts(request: Request, text: str = ""):
        """Mo's voice for one sentence, so the call page can play him."""
        import speech

        if too_fast(request, "tts", times=60) or not text.strip():
            return Response(status_code=204)
        voice = speech.Voice()
        try:
            audio = await voice.speak(text[:300], fmt="mp3_44100_64")
        except speech.SpeechUnavailable:
            return Response(status_code=204)          # the page falls back to the browser's own voice
        finally:
            await voice.aclose()
        return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "max-age=3600"})

    # ---------- M-Pesa knocking ----------
    @api.post("/mpesa/callback/{token}")
    async def mpesa_callback(request: Request, token: str):
        """Daraja saying a prompt finished. It is not believed: it only wakes us, and then we ask
        Safaricom ourselves over the authenticated API. Always 200, or Daraja retries for hours."""
        want = os.getenv("MPESA_CALLBACK_TOKEN", "")
        if not want or not hmac.compare_digest(token.encode(), want.encode()):
            return JSONResponse({"ResultCode": 0, "ResultDesc": "ignored"})
        try:
            said = mpesa.read_callback(await request.json())
        except Exception:
            return JSONResponse({"ResultCode": 0, "ResultDesc": "ignored"})
        asking = asyncio.create_task(asyncio.to_thread(_settled, said["checkout_id"]))
        _ASKING.add(asking)
        asking.add_done_callback(_ASKING.discard)
        return JSONResponse({"ResultCode": 0, "ResultDesc": "received"})

    return api
