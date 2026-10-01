"""M-Pesa payments: push a payment request to the customer's phone and hear back what they did.

Safaricom's Daraja API, the "Lipa Na M-Pesa Online" flow (STK push):

    1. we ask Daraja for a token (Basic auth with the app's key and secret),
    2. we send the push: the customer's phone rings with a PIN prompt for an exact amount,
    3. the customer enters their PIN (or doesn't),
    4. Daraja POSTs the outcome to our CallbackURL - and we can also ask it ourselves (query()).

Nothing here decides what to charge: the amount comes from the order row, never from a caller or a
browser. Sandbox credentials move no money, which is how the tests run.

Set up in .env (sandbox keys come from developer.safaricom.co.ke, free):
    MPESA_ENV=sandbox            # or production
    MPESA_CONSUMER_KEY=...
    MPESA_CONSUMER_SECRET=...
    MPESA_SHORTCODE=174379       # the sandbox till
    MPESA_PASSKEY=...
    MPESA_CALLBACK_URL=https://<your tunnel>/mpesa/callback
    KES_PER_USD=130              # the shop keeps its prices in dollars; M-Pesa charges shillings
"""
import base64
import math
import os
import re
import time
from datetime import datetime

try:
    # Antivirus HTTPS scanning re-signs TLS on this machine; verify against the Windows store.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import httpx

HOSTS = {"sandbox": "https://sandbox.safaricom.co.ke", "production": "https://api.safaricom.co.ke"}
TIMEOUT = 20.0

# What Daraja says came back. 0 is paid; the rest are the ways a payment does not happen, in words a
# shop assistant can say out loud.
OUTCOMES = {
    0: "paid",
    1: "not enough money in the M-Pesa account",
    1001: "another payment was already in progress on that phone",
    1019: "the request took too long",
    1032: "cancelled on the phone",
    1037: "no answer from the phone (it may be off, or the prompt timed out)",
    2001: "the PIN was wrong",
}


class MpesaUnavailable(Exception):
    """M-Pesa could not be asked (no keys, no network, Daraja refused). The order still stands."""


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def configured() -> bool:
    """Is there enough in .env to ask M-Pesa for anything at all?"""
    return all(env(k) for k in ("MPESA_CONSUMER_KEY", "MPESA_CONSUMER_SECRET", "MPESA_SHORTCODE",
                                "MPESA_PASSKEY"))


def host() -> str:
    return HOSTS.get(env("MPESA_ENV", "sandbox"), HOSTS["sandbox"])


def phone_number(raw: str) -> str:
    """A Kenyan number as M-Pesa wants it: 2547xxxxxxxx. Accepts 07xx, +2547xx, 7xx and spaces.

    Raises ValueError rather than guessing: a payment prompt must go to the right phone.
    """
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("0"):
        digits = "254" + digits[1:]
    elif digits.startswith("7") or digits.startswith("1"):
        digits = "254" + digits
    if not re.fullmatch(r"254[17]\d{8}", digits):
        raise ValueError(f"not a Kenyan mobile number: {raw!r}")
    return digits


def shillings(dollars: float) -> int:
    """The shop prices in dollars; M-Pesa moves whole shillings. Round up - never undercharge."""
    rate = float(env("KES_PER_USD", "130") or 130)
    return max(1, math.ceil(float(dollars) * rate))


def _password(timestamp: str) -> str:
    """Daraja's per-request password: the till, the passkey and the timestamp, base64'd."""
    raw = f"{env('MPESA_SHORTCODE')}{env('MPESA_PASSKEY')}{timestamp}"
    return base64.b64encode(raw.encode()).decode()


def outcome(result_code: int, checkout_id: str, said: str = "", receipt: str = "",
            amount: int = 0, phone: str = "") -> dict:
    """One settled payment, in the shape the rest of the shop uses."""
    return {"checkout_id": checkout_id, "settled": True, "paid": result_code == 0,
            "code": result_code, "receipt": receipt, "amount": amount, "phone": phone,
            "reason": OUTCOMES.get(result_code, said or f"M-Pesa code {result_code}")}


def read_callback(body: dict) -> dict:
    """What Daraja POSTs to MPESA_CALLBACK_URL, in the same shape as query().

    Daraja nests it: Body.stkCallback.{CheckoutRequestID, ResultCode, ResultDesc, CallbackMetadata}.
    The metadata items are a list of {Name, Value} - present on success, absent on failure.
    """
    call = ((body or {}).get("Body") or {}).get("stkCallback") or {}
    items = {i.get("Name"): i.get("Value") for i in (call.get("CallbackMetadata") or {}).get("Item", [])}
    try:
        code = int(call.get("ResultCode"))
    except (TypeError, ValueError):
        raise ValueError("not an M-Pesa callback")
    return outcome(code, str(call.get("CheckoutRequestID") or ""), str(call.get("ResultDesc") or ""),
                   receipt=str(items.get("MpesaReceiptNumber") or ""),
                   amount=int(float(items.get("Amount") or 0)),
                   phone=str(items.get("PhoneNumber") or ""))


class Mpesa:
    """A thin Daraja client. One per process is plenty; it keeps the token until it expires."""

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=TIMEOUT)
        self._token = ""
        self._token_until = 0.0

    def token(self) -> str:
        """A bearer token, cached until a minute before Daraja expires it."""
        if self._token and time.time() < self._token_until:
            return self._token
        if not configured():
            raise MpesaUnavailable("M-Pesa keys are not in .env (MPESA_CONSUMER_KEY and friends).")
        try:
            r = self.client.get(f"{host()}/oauth/v1/generate",
                                params={"grant_type": "client_credentials"},
                                auth=(env("MPESA_CONSUMER_KEY"), env("MPESA_CONSUMER_SECRET")))
        except httpx.HTTPError as e:
            raise MpesaUnavailable(f"M-Pesa is unreachable ({type(e).__name__}).") from e
        if r.status_code in (400, 401, 403):
            raise MpesaUnavailable("M-Pesa refused the keys in .env (check the consumer key and secret).")
        if r.status_code >= 400:
            raise MpesaUnavailable(f"M-Pesa would not give a token (HTTP {r.status_code}).")
        body = r.json()
        self._token = body.get("access_token") or ""
        if not self._token:
            raise MpesaUnavailable("M-Pesa returned no token.")
        self._token_until = time.time() + max(60.0, float(body.get("expires_in", 3599)) - 60)
        return self._token

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self.client.post(f"{host()}{path}", json=payload,
                                 headers={"Authorization": f"Bearer {self.token()}"})
        except httpx.HTTPError as e:
            raise MpesaUnavailable(f"M-Pesa is unreachable ({type(e).__name__}).") from e
        try:
            body = r.json()
        except Exception:
            raise MpesaUnavailable(f"M-Pesa sent something that is not JSON (HTTP {r.status_code}).")
        if r.status_code >= 400:
            # Daraja puts the real reason in errorMessage; its HTTP codes alone say little.
            raise MpesaUnavailable(body.get("errorMessage") or f"M-Pesa said no (HTTP {r.status_code}).")
        return body

    def push(self, phone: str, amount_kes: int, reference: str, description: str) -> dict:
        """Ring that phone for that amount. Returns Daraja's ids; no money has moved yet.

        reference is what the customer sees against the payment - keep it short (the order number).
        """
        callback = env("MPESA_CALLBACK_URL")
        if not callback.startswith("https://"):
            raise MpesaUnavailable("MPESA_CALLBACK_URL in .env must be a public https address "
                                  "(the cloudflared tunnel URL, then /mpesa/callback).")
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        body = self._post("/mpesa/stkpush/v1/processrequest", {
            "BusinessShortCode": env("MPESA_SHORTCODE"),
            "Password": _password(stamp),
            "Timestamp": stamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount_kes),
            "PartyA": phone_number(phone),
            "PartyB": env("MPESA_SHORTCODE"),
            "PhoneNumber": phone_number(phone),
            "CallBackURL": callback,
            "AccountReference": reference[:12],
            "TransactionDesc": description[:13],
        })
        if str(body.get("ResponseCode", "")) != "0":
            raise MpesaUnavailable(body.get("ResponseDescription") or "M-Pesa would not send the prompt.")
        return {"checkout_id": body.get("CheckoutRequestID", ""),
                "merchant_id": body.get("MerchantRequestID", ""),
                "message": body.get("CustomerMessage", "")}

    def query(self, checkout_id: str) -> dict:
        """Ask what became of a push, for when the callback never arrives. Same shape as a callback."""
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        body = self._post("/mpesa/stkpushquery/v1/query", {
            "BusinessShortCode": env("MPESA_SHORTCODE"),
            "Password": _password(stamp),
            "Timestamp": stamp,
            "CheckoutRequestID": checkout_id,
        })
        code = body.get("ResultCode")
        # While the customer is still deciding, Daraja answers "transaction is being processed":
        # that is not an outcome, it is "not yet".
        if code is None or str(code) == "":
            return {"checkout_id": checkout_id, "settled": False, "paid": False,
                    "reason": body.get("ResultDesc") or "still waiting on the phone"}
        return outcome(int(code), checkout_id, body.get("ResultDesc") or "")

    def close(self):
        self.client.close()
