"""SQLite replacement for the RDS MySQL database (same three tables)."""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "shoes.db")

# Daraja's sandbox test number. Every seed customer uses it so STK Push works out of the box,
# and because a real number in the database ends up in prompts sent to the model.
TEST_MSISDN = "254708374149"

SCHEMA = """
CREATE TABLE IF NOT EXISTS CustomerInfo (
    CustomerID        INTEGER PRIMARY KEY,
    CustomerName      TEXT NOT NULL,
    Email             TEXT,
    Phone             TEXT,
    Addr1             TEXT,
    City              TEXT,
    PreferredActivity TEXT,
    ShoeSize          REAL,
    -- Where this customer came from, now that they can sign themselves up: 'seed' for the three the
    -- shop started with, otherwise the channel that created them ('web', 'whatsapp', 'phone').
    Source            TEXT NOT NULL DEFAULT 'seed',
    CreatedAt         TEXT
);
CREATE TABLE IF NOT EXISTS ShoeInventory (
    ShoeID          INTEGER PRIMARY KEY,
    BestFitActivity TEXT,
    StyleDesc       TEXT,
    ShoeColors      TEXT,
    Price           REAL,
    InvCount        INTEGER NOT NULL CHECK (InvCount >= 0),
    -- Lowest price this pair may ever be sold for. Never sent to the model, never in a prompt:
    -- a number the model knows is a number it can be talked into saying. 0 = not negotiable,
    -- which is also the migration default, so an unseeded shoe cannot be discounted at all.
    FloorPrice      REAL NOT NULL DEFAULT 0
);
-- One row per price the shop has named. The model proposes nothing: it passes the customer's
-- number to evaluate_offer, and the price written here by the policy is the only one it may say.
CREATE TABLE IF NOT EXISTS PriceQuote (
    QuoteID        INTEGER PRIMARY KEY AUTOINCREMENT,
    SessionID      TEXT NOT NULL,
    CustomerID     INTEGER NOT NULL REFERENCES CustomerInfo(CustomerID),
    ShoeID         INTEGER NOT NULL REFERENCES ShoeInventory(ShoeID),
    ListPrice      REAL NOT NULL,
    OfferedPrice   REAL NOT NULL,
    CustomerOffer  REAL,
    Round          INTEGER NOT NULL DEFAULT 1,
    Verdict        TEXT NOT NULL,
    Status         TEXT NOT NULL DEFAULT 'OPEN',
    CreatedTurnID  TEXT,
    AcceptedTurnID TEXT,
    CreatedAt      TEXT NOT NULL,
    ExpiresAt      TEXT NOT NULL,
    OrderID        INTEGER REFERENCES OrderDetails(OrderID)
);
CREATE TABLE IF NOT EXISTS OrderDetails (
    OrderID           INTEGER PRIMARY KEY AUTOINCREMENT,
    OrderDate         TEXT NOT NULL,
    ShoeID            INTEGER NOT NULL REFERENCES ShoeInventory(ShoeID),
    CustomerID        INTEGER NOT NULL REFERENCES CustomerInfo(CustomerID),
    Status            TEXT NOT NULL DEFAULT 'PLACED',
    -- What was actually charged, copied from ShoeInventory.Price when the order is placed.
    -- Storing the number (not reading Price later) keeps repricing out of past orders.
    Amount            REAL NOT NULL DEFAULT 0,
    -- What the pair would have cost at list. ListPrice - Amount is the discount, for reporting.
    ListPrice         REAL NOT NULL DEFAULT 0,
    -- The quote this price came from, when it was negotiated. NULL for a straight list-price sale.
    QuoteID           INTEGER,
    PaymentStatus     TEXT NOT NULL DEFAULT 'UNPAID',
    -- The LAST M-Pesa prompt sent for this order, and its receipt once paid. One order can be
    -- prompted more than once (the customer cancels, the phone is off), so the attempts live in
    -- Payment; these two columns are the current state, which is what the shop reads.
    CheckoutRequestID TEXT,
    MpesaReceipt      TEXT
);
-- One row per M-Pesa prompt sent: what was asked of whom, and what came back. An order can collect
-- several. Money needs a trail that survives retries, and a callback can arrive twice - the
-- CheckoutRequestID is unique, so the second copy changes nothing.
CREATE TABLE IF NOT EXISTS Payment (
    PaymentID         INTEGER PRIMARY KEY AUTOINCREMENT,
    OrderID           INTEGER NOT NULL REFERENCES OrderDetails(OrderID),
    Phone             TEXT NOT NULL,
    AmountKes         INTEGER NOT NULL,
    CheckoutRequestID TEXT UNIQUE,
    MerchantRequestID TEXT,
    Status            TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING, PAID or FAILED
    ResultCode        INTEGER,
    Receipt           TEXT,
    Reason            TEXT,
    RequestedAt       TEXT NOT NULL,
    SettledAt         TEXT
);
"""

# Indexes live apart from SCHEMA because they are created AFTER migrate(): an index on a column that
# an older database has not been given yet ("no such column: Phone") would stop it opening at all.
INDEXES = """
-- Which customer a browser has said it is. The cookie carries only a signed random id, so a stolen
-- or forged cookie names nobody: the link between a browser and a customer lives here.
CREATE TABLE IF NOT EXISTS WebSession (
    SID        TEXT PRIMARY KEY,
    CustomerID INTEGER NOT NULL REFERENCES CustomerInfo(CustomerID),
    SeenAt     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS Payment_Order ON Payment(OrderID);
-- One prompt at a time per order, enforced by the database rather than by a check-then-write that
-- two taps can slip between. A second prompt for an order that is still pending cannot be recorded,
-- so it is never sent, so nobody is charged twice.
CREATE UNIQUE INDEX IF NOT EXISTS Payment_OnePending ON Payment(OrderID) WHERE Status = 'PENDING';
-- WhatsApp messages already answered. Meta redelivers a webhook until it gets a 200, and answering
-- the same message twice could put a second payment prompt on somebody's phone.
CREATE TABLE IF NOT EXISTS WaSeen (
    MessageID TEXT PRIMARY KEY,
    SeenAt    TEXT NOT NULL
);
-- Customers now sign themselves up (on the web, over WhatsApp, on a call), so the phone number is
-- how a returning one is found. Not unique: the seed customers share Daraja's test number, and a
-- household can share a phone. create_customer matches on the phone AND the name.
CREATE INDEX IF NOT EXISTS CustomerInfo_Phone ON CustomerInfo(Phone);
"""

# Columns added after the first version of SCHEMA, as (table, column, definition).
# CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so a shoes.db
# created before these columns existed only gets them through ALTER TABLE.
# SQLite cannot add a NOT NULL column without a default, hence the defaults below.
MIGRATIONS = [
    ("CustomerInfo", "Phone", "TEXT"),
    ("OrderDetails", "Amount", "REAL NOT NULL DEFAULT 0"),
    ("OrderDetails", "PaymentStatus", "TEXT NOT NULL DEFAULT 'UNPAID'"),
    ("OrderDetails", "CheckoutRequestID", "TEXT"),
    ("OrderDetails", "MpesaReceipt", "TEXT"),
    ("ShoeInventory", "FloorPrice", "REAL NOT NULL DEFAULT 0"),
    ("OrderDetails", "QuoteID", "INTEGER"),
    ("OrderDetails", "ListPrice", "REAL NOT NULL DEFAULT 0"),
    # Where a customer came from and when, now that they can sign themselves up: 'seed' for the
    # three the shop started with, otherwise the channel that created them.
    ("CustomerInfo", "Source", "TEXT NOT NULL DEFAULT 'seed'"),
    ("CustomerInfo", "CreatedAt", "TEXT"),
]

# CustomerID, CustomerName, Email, Phone, Addr1, City, PreferredActivity, ShoeSize
CUSTOMERS = [
    (1, "Jane Doe", "nyamungaian@gmail.com", TEST_MSISDN, "12 Park Rd", "Nairobi", "Running", 7.5),
    (2, "John Kamau", "john@example.com", TEST_MSISDN, "44 Lake Ave", "Nakuru", "Hiking", 10),
    (3, "Amina Otieno", "amina@example.com", TEST_MSISDN, "9 Hill St", "Kisumu", "Basketball", 8),
]
# ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount, FloorPrice (~80% of list)
SHOES = [
    (101, "Running", "Lightweight road runner", "Black, Blue", 89.99, 5, 72),
    (102, "Running", "Cushioned trail runner", "Grey, Orange", 119.99, 2, 95),
    (103, "Hiking", "Waterproof mid boot", "Brown", 149.99, 3, 120),
    (104, "Basketball", "High-top court shoe", "White, Red", 129.99, 0, 105),
    (105, "Walking", "Everyday comfort sneaker", "Beige, Navy", 59.99, 8, 48),
]


@contextmanager
def db():
    """Open a connection, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # One process now serves a website, WhatsApp, M-Pesa's callback and phone calls at once, so
    # reads and writes really do overlap. WAL lets readers carry on while one writer works, and the
    # busy timeout waits for that writer instead of failing the request with "database is locked".
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
        conn.close()
    except Exception:
        conn.rollback()
        conn.close()
        raise


def migrate(conn) -> list[str]:
    """Add whatever columns this database is missing. Idempotent, so it runs on every start."""
    added = []
    for table, column, definition in MIGRATIONS:
        # Table and column names cannot be bound as parameters; these come from MIGRATIONS, not input.
        if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            added.append(f"{table}.{column}")
    return added


def init_db(reset: bool = False) -> list[str]:
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with db() as conn:
        conn.executescript(SCHEMA)
        added = migrate(conn)
        conn.executescript(INDEXES)      # after migrate: they index columns migrate may have added
        conn.executemany(
            """INSERT OR IGNORE INTO CustomerInfo
               (CustomerID, CustomerName, Email, Phone, Addr1, City, PreferredActivity, ShoeSize)
               VALUES (?,?,?,?,?,?,?,?)""",
            CUSTOMERS,
        )
        conn.executemany(
            """INSERT OR IGNORE INTO ShoeInventory
               (ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount, FloorPrice)
               VALUES (?,?,?,?,?,?,?)""",
            SHOES,
        )
        # INSERT OR IGNORE skips rows that already exist, so seed rows that predate a column keep
        # its default. Fill those in - but ONLY on the start that added the column. Afterwards a
        # blank is somebody's decision: FloorPrice=0 means "not negotiable", and quietly restoring
        # the seed floor on every restart would reopen haggling the owner had switched off.
        if "CustomerInfo.Phone" in added:
            conn.executemany(
                "UPDATE CustomerInfo SET Phone = ? WHERE CustomerID = ? AND Phone IS NULL",
                [(c[3], c[0]) for c in CUSTOMERS],  # (Phone, CustomerID)
            )
        if "ShoeInventory.FloorPrice" in added:
            conn.executemany(
                "UPDATE ShoeInventory SET FloorPrice = ? WHERE ShoeID = ? AND FloorPrice <= 0",
                [(s[6], s[0]) for s in SHOES],  # (FloorPrice, ShoeID)
            )
    if added:
        print(f"  migrated {DB_PATH}: added {', '.join(added)}")
    return added


if __name__ == "__main__":
    init_db(reset=True)
    print(f"Seeded {DB_PATH}")
