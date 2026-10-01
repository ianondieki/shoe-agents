"""Imported FIRST by every test: a test must never send real email, or write to the real outbox.

Two leaks this closes, both of which happened:
  - tests/smoke_voice.py ran with the Gmail credentials from .env still active, so live test runs
    sent real confirmation emails.
  - the suites that did blank Gmail fell back to dry-run, which wrote every test order's
    confirmation into the project's real ./outbox - 84 files the owner never asked for.

Environment variables set here win over .env, because load_dotenv never overrides a variable that
is already set. That is why this has to be imported before anything else from the project.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

os.environ["SMTP_USER"] = ""
os.environ["SMTP_PASSWORD"] = ""
os.environ["OUTBOX_DIR"] = os.path.join(_HERE, "_tmp", "outbox")

REAL_OUTBOX = os.path.join(os.path.dirname(_HERE), "outbox")


def real_outbox_snapshot() -> set[str]:
    """The files currently in the project's real outbox, to prove a test left it untouched."""
    return set(os.listdir(REAL_OUTBOX)) if os.path.isdir(REAL_OUTBOX) else set()
