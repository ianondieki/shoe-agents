"""The M-Pesa client against a fake Daraja: no account, no network, no money.

Everything Safaricom's API asks of us is checked here - the password it expects, the number format,
the ids it hands back - plus every way a payment fails, because those are what a customer actually
hits: a cancelled prompt, a phone that never answers, a wrong PIN.
"""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

os.environ.update(MPESA_ENV="sandbox", MPESA_CONSUMER_KEY="key", MPESA_CONSUMER_SECRET="secret",
                  MPESA_SHORTCODE="174379", MPESA_PASSKEY="passkey",
                  MPESA_CALLBACK_URL="https://example.trycloudflare.com/mpesa/callback",
                  KES_PER_USD="130")

import httpx  # noqa: E402

import mpesa  # noqa: E402


class Daraja:
    """Safaricom's sandbox, as far as this client can tell. Records what it was sent."""

    def __init__(self):
        self.seen: list[tuple[str, dict]] = []
        self.tokens = 0
        self.push_response = {"MerchantRequestID": "29115-34620561-1",
                              "CheckoutRequestID": "ws_CO_191220191020363925", "ResponseCode": "0",
                              "ResponseDescription": "Success. Request accepted for processing",
                              "CustomerMessage": "Success. Request accepted for processing"}
        self.query_response = {"ResponseCode": "0", "ResultCode": "0",
                               "ResultDesc": "The service request is processed successfully."}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/oauth/v1/generate"):
            self.tokens += 1
            assert request.headers["authorization"].startswith("Basic "), "token needs Basic auth"
            return httpx.Response(200, json={"access_token": f"tok{self.tokens}", "expires_in": "3599"})
        body = json.loads(request.content)
        self.seen.append((path, body))
        assert request.headers["authorization"] == f"Bearer tok{self.tokens}", "calls need the token"
        if path.endswith("/stkpush/v1/processrequest"):
            return httpx.Response(200, json=self.push_response)
        if path.endswith("/stkpushquery/v1/query"):
            return httpx.Response(200, json=self.query_response)
        raise AssertionError(f"unexpected call to {path}")


def client(daraja: Daraja) -> mpesa.Mpesa:
    return mpesa.Mpesa(httpx.Client(transport=httpx.MockTransport(daraja.handler)))


# ---------- 1. a Kenyan number, however the customer writes it ----------
for written, wanted in (("0712345678", "254712345678"), ("+254712345678", "254712345678"),
                        ("254712345678", "254712345678"), ("712345678", "254712345678"),
                        ("0110 123 456", "254110123456"), ("+254 (712) 345-678", "254712345678")):
    assert mpesa.phone_number(written) == wanted, (written, mpesa.phone_number(written))
for nonsense in ("", "12345", "0812345678", "254812345678", "07123456789", "hello"):
    try:
        mpesa.phone_number(nonsense)
        raise AssertionError(f"{nonsense!r} was taken for a phone number")
    except ValueError:
        pass
print("1. phone numbers: 07xx, +254, 254, 7xx and 01xx all become 2547xxxxxxxx; nonsense is refused")

# ---------- 2. dollars to whole shillings, always rounded up ----------
assert mpesa.shillings(108) == 14040, mpesa.shillings(108)        # 108 x 130
assert mpesa.shillings(119.99) == 15599, mpesa.shillings(119.99)  # 15598.7 -> never undercharge
assert mpesa.shillings(0.001) == 1, "a payment is never zero shillings"
print(f"2. money: 108 dollars -> {mpesa.shillings(108):,} shillings, 119.99 -> "
      f"{mpesa.shillings(119.99):,} (rounded up)")

# ---------- 3. the push carries exactly what Daraja asks for ----------
d = Daraja()
m = client(d)
out = m.push("0712345678", mpesa.shillings(108), reference="Order 7", description="Shoes")
assert out["checkout_id"] == "ws_CO_191220191020363925" and out["merchant_id"], out
path, sent = d.seen[-1]
stamp = sent["Timestamp"]
assert len(stamp) == 14 and stamp.isdigit(), stamp
assert base64.b64decode(sent["Password"]).decode() == f"174379passkey{stamp}", "wrong password recipe"
assert sent["BusinessShortCode"] == "174379" and sent["PartyB"] == "174379"
assert sent["TransactionType"] == "CustomerPayBillOnline"
assert sent["Amount"] == 14040 and isinstance(sent["Amount"], int), sent["Amount"]
assert sent["PartyA"] == sent["PhoneNumber"] == "254712345678"
assert sent["CallBackURL"].startswith("https://"), sent["CallBackURL"]
assert len(sent["AccountReference"]) <= 12 and len(sent["TransactionDesc"]) <= 13
print("3. the push: whole shillings, the phone in 254 form, the password base64(till+passkey+stamp)")

# ---------- 4. one token, reused until it expires ----------
m.push("0712345678", 10, "Order 7", "Shoes")
m.query("ws_CO_191220191020363925")
assert d.tokens == 1, f"asked for {d.tokens} tokens; one should serve the whole call"
m._token_until = 0                                     # as if an hour had passed
m.query("ws_CO_191220191020363925")
assert d.tokens == 2, "an expired token must be replaced"
print("4. tokens: fetched once, reused, and replaced when they expire")

# ---------- 5. the callback, success and every failure a customer meets ----------
paid = mpesa.read_callback({"Body": {"stkCallback": {
    "MerchantRequestID": "29115-34620561-1", "CheckoutRequestID": "ws_CO_191220191020363925",
    "ResultCode": 0, "ResultDesc": "The service request is processed successfully.",
    "CallbackMetadata": {"Item": [{"Name": "Amount", "Value": 14040.0},
                                  {"Name": "MpesaReceiptNumber", "Value": "SJK2ABC123"},
                                  {"Name": "TransactionDate", "Value": 20261001121314},
                                  {"Name": "PhoneNumber", "Value": 254712345678}]}}}})
assert paid["paid"] and paid["settled"] and paid["receipt"] == "SJK2ABC123", paid
assert paid["amount"] == 14040 and paid["phone"] == "254712345678", paid
assert paid["checkout_id"] == "ws_CO_191220191020363925"

for code, expect in ((1032, "cancelled"), (1037, "no answer"), (1, "not enough money"), (2001, "PIN")):
    said = mpesa.read_callback({"Body": {"stkCallback": {
        "CheckoutRequestID": "ws_CO_1", "ResultCode": code, "ResultDesc": "whatever Daraja says"}}})
    assert said["settled"] and not said["paid"], said
    assert expect in said["reason"], (code, said["reason"])
    assert not said["receipt"], "a failed payment has no receipt"
try:
    mpesa.read_callback({"Body": {"stkCallback": {"CheckoutRequestID": "x"}}})
    raise AssertionError("a callback with no ResultCode was accepted")
except ValueError:
    pass
print("5. callbacks: a receipt on success; cancelled / no answer / no money / wrong PIN each named")

# ---------- 6. asking Daraja ourselves, for when the callback never arrives ----------
d.query_response = {"ResponseCode": "0", "ResultCode": "1032", "ResultDesc": "Request cancelled by user"}
said = m.query("ws_CO_191220191020363925")
assert said["settled"] and not said["paid"] and "cancelled" in said["reason"], said
d.query_response = {"ResponseCode": "0", "errorMessage": "transaction is being processed"}
waiting = m.query("ws_CO_191220191020363925")
assert not waiting["settled"] and not waiting["paid"], waiting
print("6. query: the outcome when there is one, and 'not yet' while the phone is still ringing")

# ---------- 7. nothing works without keys, and nothing is guessed ----------
d.push_response = {"ResponseCode": "1", "ResponseDescription": "Invalid Amount"}
try:
    m.push("0712345678", 0, "Order 7", "Shoes")
    raise AssertionError("a refused push was treated as sent")
except mpesa.MpesaUnavailable as e:
    assert "Invalid Amount" in str(e), e

for missing in ("MPESA_CONSUMER_KEY", "MPESA_PASSKEY"):
    keep, os.environ[missing] = os.environ[missing], ""
    try:
        assert not mpesa.configured(), f"{missing} empty should count as not configured"
        try:
            client(Daraja()).token()
            raise AssertionError("asked Daraja for a token with no keys")
        except mpesa.MpesaUnavailable as e:
            assert ".env" in str(e), e
    finally:
        os.environ[missing] = keep

os.environ["MPESA_CALLBACK_URL"] = "http://localhost:8013/mpesa/callback"
try:
    client(Daraja()).push("0712345678", 10, "Order 7", "Shoes")
    raise AssertionError("pushed with a callback URL M-Pesa cannot reach")
except mpesa.MpesaUnavailable as e:
    assert "https" in str(e), e
os.environ["MPESA_CALLBACK_URL"] = "https://example.trycloudflare.com/mpesa/callback"
print("7. refusals: a rejected push, missing keys and a non-public callback URL all say what to fix")

print("\nALL MPESA ASSERTIONS PASSED")
