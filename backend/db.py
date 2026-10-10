"""Database setup and connection. SQLite file lives next to this script."""
import os
import sqlite3
from pathlib import Path

# Override with REELREAD_DB=/path/to/file.db (tests use a temp file).
DB_PATH = Path(os.environ.get("REELREAD_DB", Path(__file__).parent / "reelread.db"))

TOPICS = {            # id -> label shown in the UI. Ids match seeds/<id>.txt
    "tech": "Tech",
    "sports": "Sports",
    "ai_ml": "AI & ML",
    "finance": "Finance",
    "politics": "Politics",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id          INTEGER PRIMARY KEY,   -- Wikipedia page id
    title       TEXT NOT NULL,
    extract     TEXT NOT NULL,         -- plain text only, never HTML
    url         TEXT NOT NULL,
    thumbnail   TEXT,                  -- may be NULL; only https://upload. or thumb.wikimedia.org/...
    fetched_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- One article can belong to several topics (e.g. "Python" is Tech and AI & ML).
CREATE TABLE IF NOT EXISTS card_topics (
    card_id  INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    topic    TEXT NOT NULL,
    PRIMARY KEY (card_id, topic)
);

-- How each line of seeds/*.txt resolved: a card id, or NULL if rejected (stub,
-- disambiguation, 404). Lets fetch_cards.py rerun without re-downloading.
CREATE TABLE IF NOT EXISTS seed_titles (
    seed_title  TEXT PRIMARY KEY,
    card_id     INTEGER                -- no FK: pruned cards may leave stale ids, harmless
);

-- Anonymous users. id is a random UUID4 stored in an HttpOnly cookie.
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    topics      TEXT NOT NULL,         -- JSON list of topic ids
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Everything the ranker learns from.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    card_id     INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    type        TEXT NOT NULL CHECK (type IN ('view','like','unlike','quiz_shown','quiz_correct','quiz_wrong')),
    dwell_ms    INTEGER,               -- only for 'view'
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Feed queries filter by user and type and want the newest first (ORDER BY id DESC).
CREATE INDEX IF NOT EXISTS idx_events_user_type ON events(user_id, type, id);

-- Cards already sent to a user, so prefetching never repeats a card.
CREATE TABLE IF NOT EXISTS served (
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    card_id     INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    served_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, card_id)
);

-- One pre-built recall question per card (built offline by build_index.py).
CREATE TABLE IF NOT EXISTS quizzes (
    card_id       INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
    question      TEXT NOT NULL,       -- first sentence with the subject replaced by ___
    options       TEXT NOT NULL,       -- JSON list of 4 card titles
    answer_index  INTEGER NOT NULL CHECK (answer_index BETWEEN 0 AND 3)
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
