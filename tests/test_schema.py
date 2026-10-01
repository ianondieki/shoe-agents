"""Check the added columns and the price snapshot, on fresh and pre-existing databases."""
import _safety  # noqa: F401  - first: no real email and no writes to the real outbox, ever
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(HERE, "_tmp")
os.makedirs(SCRATCH, exist_ok=True)
DB = os.path.join(SCRATCH, "schema_test.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["DB_PATH"] = DB
os.environ["TRACE_PATH"] = ""  # no trace file for this one
sys.path.insert(0, ROOT)

# --- a database in the OLD shape, with an order already in it ---------------------------
OLD_SCHEMA = """
CREATE TABLE CustomerInfo (
    CustomerID INTEGER PRIMARY KEY, CustomerName TEXT NOT NULL, Email TEXT,
    Addr1 TEXT, City TEXT, PreferredActivity TEXT, ShoeSize REAL);
CREATE TABLE ShoeInventory (
    ShoeID INTEGER PRIMARY KEY, BestFitActivity TEXT, StyleDesc TEXT, ShoeColors TEXT,
    Price REAL, InvCount INTEGER NOT NULL CHECK (InvCount >= 0));
CREATE TABLE OrderDetails (
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT, OrderDate TEXT NOT NULL,
    ShoeID INTEGER NOT NULL REFERENCES ShoeInventory(ShoeID),
    CustomerID INTEGER NOT NULL REFERENCES CustomerInfo(CustomerID),
    Status TEXT NOT NULL DEFAULT 'PLACED');
"""
old = sqlite3.connect(DB)
old.executescript(OLD_SCHEMA)
old.execute("INSERT INTO CustomerInfo VALUES (1,'Jane Doe','jane@example.com','12 Park Rd','Nairobi','Running',7.5)")
old.execute("INSERT INTO ShoeInventory VALUES (101,'Running','Lightweight road runner','Black, Blue',89.99,5)")
old.execute("INSERT INTO OrderDetails (OrderDate, ShoeID, CustomerID) VALUES ('2026-09-01',101,1)")
old.commit()
old.close()
print("built a pre-migration database with 1 customer, 1 shoe, 1 existing order")

from db import MIGRATIONS, TEST_MSISDN, db, init_db  # noqa: E402
from tools import cancel_order, list_orders, place_order  # noqa: E402

# --- migrating it in place -------------------------------------------------------------
added = init_db()
assert added == [f"{t}.{c}" for t, c, _ in MIGRATIONS], added
print(f"migrate() added: {', '.join(added)}")

with db() as conn:
    cols = {t: {r["name"] for r in conn.execute(f"PRAGMA table_info({t})")}
            for t in ("CustomerInfo", "ShoeInventory", "OrderDetails")}
    assert "Phone" in cols["CustomerInfo"]
    assert {"Amount", "PaymentStatus", "CheckoutRequestID", "MpesaReceipt"} <= cols["OrderDetails"]
    # the pre-existing order survived, with honest defaults
    old_order = conn.execute("SELECT * FROM OrderDetails WHERE OrderID=1").fetchone()
    assert old_order["ShoeID"] == 101 and old_order["OrderDate"] == "2026-09-01", dict(old_order)
    assert old_order["Amount"] == 0 and old_order["PaymentStatus"] == "UNPAID", dict(old_order)
    # the customer that predates the Phone column got backfilled, and was not otherwise touched
    jane = conn.execute("SELECT * FROM CustomerInfo WHERE CustomerID=1").fetchone()
    assert jane["Phone"] == TEST_MSISDN and jane["CustomerName"] == "Jane Doe", dict(jane)
print("existing rows preserved; Phone backfilled; new columns defaulted")

# running it again must change nothing
assert init_db() == [], "migrate() is not idempotent"
print("migrate() is idempotent")

# --- the snapshot ----------------------------------------------------------------------
msg = place_order.invoke({"shoe_id": 101, "customer_id": 1})
assert msg == "Order 2 placed for 89.99.", msg
with db() as conn:
    assert conn.execute("SELECT Amount FROM OrderDetails WHERE OrderID=2").fetchone()["Amount"] == 89.99
    assert conn.execute("SELECT InvCount FROM ShoeInventory WHERE ShoeID=101").fetchone()["InvCount"] == 4

before = json.loads(list_orders.invoke({"customer_id": 1}))
with db() as conn:  # the shop reprices the shoe, exactly as in the demo earlier
    conn.execute("UPDATE ShoeInventory SET Price = 129.99 WHERE ShoeID = 101")
after = json.loads(list_orders.invoke({"customer_id": 1}))

assert before == after, f"repricing changed past orders\n before={before}\n after={after}"
assert [o["Amount"] for o in after] == [0, 89.99], after
assert "Price" not in after[0], "list_orders still reports the live inventory price"
print(f"after repricing 89.99 -> 129.99, order 2 still reads {after[1]['Amount']}")

# a new order picks up the new price; the old one does not
place_order.invoke({"shoe_id": 101, "customer_id": 1})
amounts = [o["Amount"] for o in json.loads(list_orders.invoke({"customer_id": 1}))]
assert amounts == [0, 89.99, 129.99], amounts
print(f"amounts after a second order at the new price: {amounts}")

# cancelling still works and restores stock
assert "cancelled" in cancel_order.invoke({"order_id": 2})
with db() as conn:
    assert conn.execute("SELECT InvCount FROM ShoeInventory WHERE ShoeID=101").fetchone()["InvCount"] == 4
    assert conn.execute("SELECT Amount FROM OrderDetails WHERE OrderID=2").fetchone()["Amount"] == 89.99

# --- and a database built fresh from SCHEMA ends up identical ---------------------------
fresh = os.path.join(SCRATCH, "fresh.db")
if os.path.exists(fresh):
    os.remove(fresh)
os.environ["DB_PATH"] = fresh
import db as db_module  # noqa: E402

db_module.DB_PATH = fresh
assert db_module.init_db(reset=True) == [], "a fresh database should need no migration"
conn = sqlite3.connect(fresh)
conn.row_factory = sqlite3.Row
for table in ("CustomerInfo", "ShoeInventory", "OrderDetails"):
    # Every table is compared - skipping one silently is how a column added to SCHEMA but not to
    # MIGRATIONS (so missing from every upgraded shoes.db) slipped past this check before.
    assert table in cols, f"{table} was not captured from the migrated database"
    fresh_cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    assert fresh_cols == cols[table], f"{table}: fresh {fresh_cols} != migrated {cols[table]}"
assert conn.execute("SELECT COUNT(*) c FROM CustomerInfo WHERE Phone IS NULL").fetchone()["c"] == 0
conn.close()
print("a fresh database has the same columns as a migrated one")

print("\nALL SCHEMA ASSERTIONS PASSED")
