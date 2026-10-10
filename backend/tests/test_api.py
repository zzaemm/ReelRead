"""API tests for main.py + security.py.

Every test gets a brand-new temporary SQLite file with a small set of synthetic
cards, so nothing here touches backend/reelread.db or the network.

Run from backend/:  python3 -m pytest tests/test_api.py -q
"""
import json
import os
import sys
import uuid
from pathlib import Path

# Point the app at throwaway paths BEFORE importing db/main, so even an
# accidental connect() at import time could never hit the real database.
_SCRATCH = Path(__file__).resolve().parent / ".nonexistent"
os.environ["REELREAD_DB"] = str(_SCRATCH / "never-used.db")
os.environ["REELREAD_INDEX"] = str(_SCRATCH / "no-index.npz")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402
import security  # noqa: E402

# --- synthetic data --------------------------------------------------------
TECH = list(range(1, 26))          # 25 tech cards; card 1 is also ai_ml
SPORTS = list(range(101, 106))     # 5 sports cards
POLITICS = [201, 202, 203]         # 3 politics cards; 201 has a quiz
FINANCE = [301, 302, 303, 304]     # 301-303 have quizzes, 304 doesn't
QUIZ_CARDS = {201: 0, 301: 2, 302: 1, 303: 3}   # card_id -> answer_index


def seed(conn):
    def add(cid, topics):
        conn.execute(
            "INSERT INTO cards (id, title, extract, url, thumbnail) VALUES (?, ?, ?, ?, ?)",
            (cid, f"Card {cid}", f"Card {cid} is a synthetic test article.",
             f"https://en.wikipedia.org/wiki/Card_{cid}",
             "https://upload.wikimedia.org/x.jpg" if cid % 2 else None),
        )
        for t in topics:
            conn.execute("INSERT INTO card_topics (card_id, topic) VALUES (?, ?)", (cid, t))

    for cid in TECH:
        add(cid, ["tech", "ai_ml"] if cid == 1 else ["tech"])
    for cid in SPORTS:
        add(cid, ["sports"])
    for cid in POLITICS:
        add(cid, ["politics"])
    for cid in FINANCE:
        add(cid, ["finance"])
    for cid, answer in QUIZ_CARDS.items():
        conn.execute(
            "INSERT INTO quizzes (card_id, question, options, answer_index) VALUES (?, ?, ?, ?)",
            (cid, f"___ is question {cid}.", json.dumps(["A", "B", "C", "D"]), answer),
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("REELREAD_INDEX", str(tmp_path / "missing-index.npz"))
    monkeypatch.delenv("REELREAD_SECURE_COOKIE", raising=False)
    security.reset_rate_limits()
    with TestClient(main.app) as c:      # `with` runs startup (init_db, Ranker.load)
        conn = db.connect()
        with conn:
            seed(conn)
        conn.close()
        yield c
    security.reset_rate_limits()


def query(sql, params=()):
    conn = db.connect()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def start_session(client, topics):
    r = client.post("/api/session", json={"topics": topics})
    assert r.status_code == 200, r.text
    return client.cookies.get("rr_uid")


def set_uid_cookie(client, value):
    """Set rr_uid by hand, on the same domain the test client files server cookies under."""
    client.cookies.clear()
    client.cookies.set("rr_uid", value, domain="testserver.local")


def send_events(client, events):
    return client.post("/api/events", json={"events": events})


def feed(client, n=10):
    r = client.get("/api/feed", params={"n": n})
    assert r.status_code == 200, r.text
    return r.json()["items"]


def card_ids(items):
    return [it["card"]["id"] for it in items if it["kind"] == "card"]


def quizzes(items):
    return [it for it in items if it["kind"] == "quiz"]


# --- topics ----------------------------------------------------------------

def test_topics_counts(client):
    r = client.get("/api/topics")
    assert r.status_code == 200
    data = {t["id"]: t for t in r.json()}
    assert [t["id"] for t in r.json()] == list(db.TOPICS)
    assert data["tech"] == {"id": "tech", "label": "Tech", "count": 25}
    assert data["ai_ml"]["count"] == 1 and data["ai_ml"]["label"] == "AI & ML"
    assert data["sports"]["count"] == 5
    assert data["politics"]["count"] == 3
    assert data["finance"]["count"] == 4


# --- session / cookie ------------------------------------------------------

def test_session_creates_user_and_sets_cookie(client):
    r = client.post("/api/session", json={"topics": ["tech", "ai_ml"]})
    assert r.status_code == 200
    assert r.json() == {"topics": ["tech", "ai_ml"]}
    uid = client.cookies.get("rr_uid")
    assert security.is_valid_uuid4(uid)
    cookie = r.headers["set-cookie"].lower()
    for flag in ("httponly", "samesite=lax", "path=/", "max-age=31536000"):
        assert flag in cookie
    assert "secure" not in cookie
    assert client.get("/api/me").json() == {"topics": ["tech", "ai_ml"]}


def test_secure_cookie_flag_from_env(client, monkeypatch):
    monkeypatch.setenv("REELREAD_SECURE_COOKIE", "1")
    r = client.post("/api/session", json={"topics": ["tech"]})
    assert "secure" in r.headers["set-cookie"].lower()


def test_session_update_keeps_same_user(client):
    uid = start_session(client, ["tech"])
    r = client.post("/api/session", json={"topics": ["sports", "finance"]})
    assert r.status_code == 200
    assert client.cookies.get("rr_uid") == uid
    assert len(query("SELECT * FROM users")) == 1
    assert client.get("/api/me").json() == {"topics": ["sports", "finance"]}


def test_session_with_unknown_cookie_creates_new_user(client):
    set_uid_cookie(client, str(uuid.uuid4()))
    start_session(client, ["tech"])
    assert len(query("SELECT * FROM users")) == 1
    assert client.get("/api/me").status_code == 200


@pytest.mark.parametrize("topics", [
    [], ["cooking"], ["tech", "tech"], ["tech", "sports", "ai_ml", "finance", "politics", "tech"],
    [1], "tech",
])
def test_session_invalid_topics_422(client, topics):
    r = client.post("/api/session", json={"topics": topics})
    assert r.status_code == 422
    assert "set-cookie" not in r.headers


def test_session_all_five_topics_ok(client):
    start_session(client, list(db.TOPICS))


# --- 401 paths -------------------------------------------------------------

@pytest.mark.parametrize("cookie", [
    None,                                   # no cookie
    "not-a-uuid",                           # malformed
    "'; DROP TABLE users; --",              # malformed (and nasty)
    str(uuid.uuid1()),                      # wrong UUID version
    str(uuid.uuid4()).upper(),              # non-canonical
    str(uuid.uuid4()),                      # well-formed but unknown
])
def test_user_routes_401(client, cookie):
    if cookie is not None:
        set_uid_cookie(client, cookie)
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/feed").status_code == 401
    assert send_events(client, []).status_code == 401
    assert client.post("/api/quiz/answer", json={"card_id": 201, "choice": 0}).status_code == 401


# --- feed ------------------------------------------------------------------

def test_feed_returns_n_distinct_cards_from_chosen_topics(client):
    start_session(client, ["sports", "politics"])
    items = feed(client, n=6)
    ids = card_ids(items)
    assert len(ids) == 6 and len(set(ids)) == 6
    assert set(ids) <= set(SPORTS + POLITICS)
    card = items[0]["card"]
    assert set(card) == {"id", "title", "extract", "url", "thumbnail", "topics", "liked"}
    assert card["liked"] is False


def test_feed_card_shape_topics_and_thumbnail(client):
    start_session(client, ["ai_ml"])
    (item,) = feed(client, n=5)            # ai_ml has only card 1
    assert item["kind"] == "card"
    card = item["card"]
    assert card["id"] == 1
    assert card["topics"] == ["ai_ml", "tech"]
    assert card["thumbnail"] == "https://upload.wikimedia.org/x.jpg"
    assert card["url"] == "https://en.wikipedia.org/wiki/Card_1"


def test_feed_no_repeats_until_exhausted_then_resets(client):
    uid = start_session(client, ["tech"])
    first = card_ids(feed(client, n=10))
    second = card_ids(feed(client, n=10))
    assert len(first) == 10 and len(second) == 10
    assert not set(first) & set(second)
    # 5 left < 10 requested -> served is cleared and we start over.
    third = card_ids(feed(client, n=10))
    assert len(third) == 10 and set(third) <= set(TECH)
    served = {r["card_id"] for r in query("SELECT card_id FROM served WHERE user_id = ?", (uid,))}
    assert served == set(third)


def test_feed_n_validation(client):
    start_session(client, ["tech"])
    for bad in (0, 21, "abc"):
        assert client.get("/api/feed", params={"n": bad}).status_code == 422
    assert len(card_ids(feed(client, n=20))) == 20
    assert len(card_ids(client.get("/api/feed").json()["items"])) == 10   # default n


def test_feed_liked_flag(client):
    start_session(client, ["politics"])
    assert send_events(client, [
        {"card_id": 201, "type": "like"}, {"card_id": 201, "type": "like"},
        {"card_id": 201, "type": "unlike"},                                 # 2 > 1 -> liked
        {"card_id": 202, "type": "like"}, {"card_id": 202, "type": "unlike"},  # 1 = 1 -> not
    ]).status_code == 204
    cards = {it["card"]["id"]: it["card"] for it in feed(client, n=3) if it["kind"] == "card"}
    assert cards[201]["liked"] is True
    assert cards[202]["liked"] is False
    assert cards[203]["liked"] is False


# --- quiz in the feed ------------------------------------------------------

def test_quiz_only_when_eligible_and_without_answer(client):
    start_session(client, ["tech"])          # quiz cards are in finance/politics
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 3000}])
    assert quizzes(feed(client, n=10)) == []                 # 3000 ms: not yet

    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 2500}])  # total 5500
    items = feed(client, n=10)
    assert items[8]["kind"] == "quiz"                         # after the 8th card
    assert len(quizzes(items)) == 1 and len(card_ids(items)) == 10
    q = items[8]["quiz"]
    assert q == {"card_id": 301, "question": "___ is question 301.",
                 "options": ["A", "B", "C", "D"]}
    assert "answer_index" not in json.dumps(items)


def test_quiz_needs_quiz_row(client):
    start_session(client, ["tech"])
    send_events(client, [{"card_id": 304, "type": "view", "dwell_ms": 9000}])   # no quizzes row
    assert quizzes(feed(client)) == []


def test_quiz_at_end_when_fewer_than_8_cards(client):
    start_session(client, ["sports"])
    send_events(client, [{"card_id": 302, "type": "view", "dwell_ms": 6000}])
    items = feed(client, n=5)
    assert len(items) == 6 and items[-1]["kind"] == "quiz"
    assert items[-1]["quiz"]["card_id"] == 302


def test_quiz_picks_most_recently_viewed_and_skips_answered(client):
    start_session(client, ["tech"])
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 6000}])
    send_events(client, [{"card_id": 303, "type": "view", "dwell_ms": 6000}])
    assert quizzes(feed(client))[0]["quiz"]["card_id"] == 303
    client.post("/api/quiz/answer", json={"card_id": 303, "choice": 0})
    assert quizzes(feed(client))[0]["quiz"]["card_id"] == 301


def test_quiz_not_about_card_in_same_response(client):
    start_session(client, ["politics"])
    send_events(client, [{"card_id": 201, "type": "view", "dwell_ms": 6000}])
    items = feed(client, n=3)                 # contains all politics cards incl. 201
    assert 201 in card_ids(items)
    assert quizzes(items) == []


# --- quiz answers ----------------------------------------------------------

def mark_quiz_shown(uid, card_id):
    """Pretend the feed already showed this quiz (answers need a quiz_shown event)."""
    conn = db.connect()
    with conn:
        conn.execute("INSERT INTO events (user_id, card_id, type) VALUES (?, ?, 'quiz_shown')",
                     (uid, card_id))
    conn.close()


def quiz_events(uid):
    return query("SELECT type FROM events WHERE user_id = ? AND type IN ('quiz_correct', 'quiz_wrong')", (uid,))


def test_quiz_answer_correct(client):
    uid = start_session(client, ["tech"])
    mark_quiz_shown(uid, 301)
    r = client.post("/api/quiz/answer", json={"card_id": 301, "choice": 2})
    assert r.status_code == 200 and r.json() == {"correct": True, "answer_index": 2}
    assert [e["type"] for e in quiz_events(uid)] == ["quiz_correct"]


def test_quiz_answer_wrong_then_idempotent(client):
    uid = start_session(client, ["tech"])
    mark_quiz_shown(uid, 302)
    r = client.post("/api/quiz/answer", json={"card_id": 302, "choice": 3})
    assert r.json() == {"correct": False, "answer_index": 1}
    # Answering again (even correctly) returns the first result and logs nothing.
    r = client.post("/api/quiz/answer", json={"card_id": 302, "choice": 1})
    assert r.status_code == 200 and r.json() == {"correct": False, "answer_index": 1}
    assert [e["type"] for e in quiz_events(uid)] == ["quiz_wrong"]


def test_quiz_answer_404_and_422(client):
    start_session(client, ["tech"])
    assert client.post("/api/quiz/answer", json={"card_id": 304, "choice": 0}).status_code == 404
    assert client.post("/api/quiz/answer", json={"card_id": 99999, "choice": 0}).status_code == 404
    for bad in ({"card_id": 301, "choice": 4}, {"card_id": 301, "choice": -1},
                {"card_id": 301}, {"card_id": "301", "choice": 0}):
        assert client.post("/api/quiz/answer", json=bad).status_code == 422


# --- events ----------------------------------------------------------------

def test_events_stored_unknown_cards_dropped_dwell_clamped(client):
    uid = start_session(client, ["tech"])
    r = send_events(client, [
        {"card_id": 1, "type": "view", "dwell_ms": 10**9},       # clamped to 600000
        {"card_id": 2, "type": "view", "dwell_ms": -50},         # clamped to 0
        {"card_id": 3, "type": "view", "dwell_ms": 1234.6},      # rounded
        {"card_id": 4, "type": "like", "dwell_ms": 999},         # dwell ignored for likes
        {"card_id": 424242, "type": "view", "dwell_ms": 100},    # unknown card: dropped
    ])
    assert r.status_code == 204 and r.content == b""
    rows = query("SELECT card_id, type, dwell_ms FROM events WHERE user_id = ? ORDER BY id", (uid,))
    assert [tuple(r) for r in rows] == [
        (1, "view", 600000), (2, "view", 0), (3, "view", 1235), (4, "like", None)]


@pytest.mark.parametrize("payload", [
    {"events": [{"card_id": 1, "type": "quiz_correct"}]},     # clients can't send quiz_*
    {"events": [{"card_id": 1, "type": "quiz_wrong"}]},
    {"events": [{"card_id": 1, "type": "view"}]},             # view without dwell_ms
    {"events": [{"card_id": "1", "type": "like"}]},
    {"events": [{"card_id": 1, "type": "view", "dwell_ms": 1}] * 51},   # > 50 events
    {"nope": []},
])
def test_events_validation_422(client, payload):
    uid = start_session(client, ["tech"])
    assert client.post("/api/events", json=payload).status_code == 422
    assert query("SELECT * FROM events WHERE user_id = ?", (uid,)) == []


def test_events_invalid_json_422_and_50_ok(client):
    start_session(client, ["tech"])
    r = client.post("/api/events", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    ok = [{"card_id": 1, "type": "view", "dwell_ms": 1}] * 50
    assert send_events(client, ok).status_code == 204


def test_events_body_too_large_413(client):
    start_session(client, ["tech"])
    big = json.dumps({"events": [], "pad": "x" * 17000}).encode()
    r = client.post("/api/events", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_events_chunked_body_too_large_413(client):
    """No Content-Length (chunked upload): the streaming counter must still stop it."""
    start_session(client, ["tech"])

    def chunks():
        for _ in range(20):
            yield b"x" * 1024

    r = client.post("/api/events", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_events_accepts_beacon_style_body(client):
    """sendBeacon posts a Blob; there's no Origin on same-origin beacons in some browsers."""
    uid = start_session(client, ["tech"])
    body = json.dumps({"events": [{"card_id": 5, "type": "view", "dwell_ms": 2000}]})
    r = client.post("/api/events", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 204
    assert len(query("SELECT * FROM events WHERE user_id = ?", (uid,))) == 1


# --- rate limiting ---------------------------------------------------------

def test_feed_rate_limit_429(client):
    start_session(client, ["sports"])
    for _ in range(30):
        assert client.get("/api/feed", params={"n": 1}).status_code == 200
    r = client.get("/api/feed", params={"n": 1})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


def test_events_rate_limit_429(client):
    start_session(client, ["tech"])
    for _ in range(60):
        assert send_events(client, []).status_code == 204
    assert send_events(client, []).status_code == 429


def test_quiz_rate_limit_429(client):
    uid = start_session(client, ["tech"])
    mark_quiz_shown(uid, 301)
    for _ in range(30):
        assert client.post("/api/quiz/answer", json={"card_id": 301, "choice": 0}).status_code == 200
    assert client.post("/api/quiz/answer", json={"card_id": 301, "choice": 0}).status_code == 429


def test_session_creation_rate_limit_per_ip(client):
    for _ in range(10):
        client.cookies.clear()
        start_session(client, ["tech"])
    client.cookies.clear()
    assert client.post("/api/session", json={"topics": ["tech"]}).status_code == 429
    assert len(query("SELECT * FROM users")) == 10


def test_rate_limits_are_per_user(client):
    start_session(client, ["sports"])
    for _ in range(30):
        client.get("/api/feed", params={"n": 1})
    assert client.get("/api/feed", params={"n": 1}).status_code == 429
    client.cookies.clear()
    start_session(client, ["sports"])             # a different user has their own budget
    assert client.get("/api/feed", params={"n": 1}).status_code == 200


def test_rate_limiter_window_slides():
    now = [1000.0]
    limiter = security.RateLimiter(limit=2, window=60, clock=lambda: now[0])
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(60)
    assert limiter.hit("b") is None                     # keys are independent
    now[0] += 30
    assert limiter.hit("a") == pytest.approx(30)
    now[0] += 30.001
    assert limiter.hit("a") is None                     # oldest hits expired
    limiter.reset()
    assert limiter.hit("a") is None and limiter.hit("a") is None


# --- CSRF / Origin ---------------------------------------------------------

@pytest.mark.parametrize("origin", [
    "http://evil.example", "http://testserver:8080", "null", "http://testserver.evil.example",
])
def test_cross_origin_post_403(client, origin):
    r = client.post("/api/session", json={"topics": ["tech"]}, headers={"Origin": origin})
    assert r.status_code == 403
    assert "set-cookie" not in r.headers
    assert r.headers["x-frame-options"] == "DENY"
    assert query("SELECT * FROM users") == []


def test_same_origin_or_missing_origin_post_allowed(client):
    r = client.post("/api/session", json={"topics": ["tech"]},
                    headers={"Origin": "http://testserver"})
    assert r.status_code == 200
    assert send_events(client, []).status_code == 204     # no Origin header at all


def test_origin_matches_host_unit():
    f = security.origin_matches_host
    assert f("http://localhost:8000", "localhost:8000")
    assert f("http://example.com", "example.com:80")
    assert f("https://example.com", "example.com", scheme="https")
    assert f("http://LOCALHOST:8000", "localhost:8000")
    assert not f("http://localhost:8001", "localhost:8000")
    assert not f("http://evil.com", "localhost:8000")
    assert not f("http://localhost:8000", None)
    assert not f("http://localhost:abc", "localhost:8000")
    assert not f("ftp://localhost:8000", "localhost:8000")


# --- security headers ------------------------------------------------------

def assert_security_headers(r):
    for name, value in security.SECURITY_HEADERS.items():
        assert r.headers.get(name) == value, (name, r.request.url)


def test_security_headers_on_all_responses(client):
    assert_security_headers(client.get("/api/topics"))                            # 200 JSON
    assert_security_headers(client.get("/api/me"))                                # 401
    assert_security_headers(client.post("/api/session", json={"topics": []}))     # 422
    assert_security_headers(client.get("/api/does-not-exist"))                    # 404 JSON
    assert_security_headers(client.get("/"))                                      # static (or 404)
    assert_security_headers(client.get("/no-such-file.js"))                       # static 404
    assert client.get("/api/topics").headers["cache-control"] == "no-store"


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json() == {"detail": "Not found."}


def test_skipped_quiz_is_not_repeated(client):
    """A quiz that was shown but not answered must not come back on later pages."""
    start_session(client, ["tech"])
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 6000}])
    assert quizzes(feed(client))[0]["quiz"]["card_id"] == 301
    assert quizzes(feed(client)) == []          # skipped, not answered: still gone
    # ...but it can still be answered (e.g. user scrolls back up to it)
    r = client.post("/api/quiz/answer", json={"card_id": 301, "choice": 0})
    assert r.status_code == 200


# --- review fixes (regression tests) ---------------------------------------

# 1. Body size caps on every route, and 422s that don't echo the input.

def test_body_cap_4kb_on_other_routes(client):
    big = json.dumps({"topics": ["tech"], "pad": "x" * 5000}).encode()
    r = client.post("/api/session", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert_security_headers(r)
    uid = start_session(client, ["tech"])
    mark_quiz_shown(uid, 301)
    big = json.dumps({"card_id": 301, "choice": 0, "pad": "x" * 5000}).encode()
    r = client.post("/api/quiz/answer", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert quiz_events(uid) == []


def test_body_cap_chunked_without_content_length(client):
    def chunks():                       # 40 KB in 1 KB pieces, no Content-Length
        for _ in range(40):
            yield b"x" * 1024

    for path in ("/api/session", "/api/quiz/answer", "/api/me"):
        r = client.post(path, content=chunks(), headers={"Content-Type": "application/json"})
        assert r.status_code == 413, path


def test_events_cap_is_16kb_not_4kb(client):
    start_session(client, ["tech"])
    events = [{"card_id": 1, "type": "view", "dwell_ms": 123456}] * 50
    body = json.dumps({"events": events, "pad": "x" * 8000}).encode()   # extra keys ignored
    assert 4096 < len(body) <= 16384
    r = client.post("/api/events", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 204


def test_invalid_content_length_400(client):
    r = client.post("/api/session", content=b'{"topics":["tech"]}',
                    headers={"Content-Type": "application/json", "Content-Length": "abc"})
    assert r.status_code == 400


def test_422_does_not_echo_input(client):
    marker = "ECHO_MARKER_" + "y" * 200
    responses = [
        client.post("/api/session", json={"topics": [marker]}),
        client.post("/api/session", json={"topics": "tech", "x": marker}),
    ]
    start_session(client, ["tech"])
    responses += [
        send_events(client, [{"card_id": 1, "type": marker}]),
        client.post("/api/quiz/answer", json={"card_id": marker, "choice": 0}),
        client.get("/api/feed", params={"n": marker}),
    ]
    for r in responses:
        assert r.status_code == 422
        assert "ECHO_MARKER" not in r.text
        for err in r.json()["detail"]:
            assert "input" not in err and "ctx" not in err
            assert set(err) <= {"type", "loc", "msg"}


# 2. Storage caps and index use.

def insert_views(uid, card_id, count, dwell_ms=100):
    conn = db.connect()
    with conn:
        conn.executemany(
            "INSERT INTO events (user_id, card_id, type, dwell_ms) VALUES (?, ?, 'view', ?)",
            [(uid, card_id, dwell_ms)] * count)
    conn.close()


def test_events_trimmed_to_newest_2000_signals_quiz_events_kept(client):
    uid = start_session(client, ["tech"])
    mark_quiz_shown(uid, 301)                   # an old quiz event: must survive the trim
    client.post("/api/quiz/answer", json={"card_id": 301, "choice": 2})
    insert_views(uid, 1, main.MAX_STORED_SIGNALS + 100)
    assert send_events(client, [{"card_id": 2, "type": "like"}]).status_code == 204

    signals = query("SELECT id, card_id, type FROM events WHERE user_id = ? "
                    "AND type IN ('view', 'like', 'unlike') ORDER BY id", (uid,))
    assert len(signals) == main.MAX_STORED_SIGNALS
    assert signals[-1]["card_id"] == 2 and signals[-1]["type"] == "like"   # newest kept
    kept_quiz = query("SELECT type FROM events WHERE user_id = ? AND type LIKE 'quiz_%' "
                      "ORDER BY id", (uid,))
    assert [r["type"] for r in kept_quiz] == ["quiz_shown", "quiz_correct"]


def test_trim_only_affects_that_user(client):
    other = start_session(client, ["tech"])
    insert_views(other, 1, main.MAX_STORED_SIGNALS + 10)
    client.cookies.clear()
    start_session(client, ["tech"])
    send_events(client, [{"card_id": 1, "type": "like"}])
    assert len(query("SELECT 1 FROM events WHERE user_id = ?", (other,))) == \
        main.MAX_STORED_SIGNALS + 10


def test_recent_signals_uses_index_without_temp_btree(client):
    conn = db.connect()
    plan = [row[3] for row in conn.execute(
        "EXPLAIN QUERY PLAN " + main.RECENT_SIGNALS_SQL, {"uid": "u", "limit": 300})]
    conn.close()
    assert not any("TEMP B-TREE" in step for step in plan), plan
    assert any("idx_events_user_type" in step for step in plan), plan


def test_recent_signals_order_and_limit(client):
    uid = start_session(client, ["tech"])
    insert_views(uid, 1, 310)
    send_events(client, [{"card_id": 2, "type": "like"}, {"card_id": 3, "type": "unlike"},
                         {"card_id": 4, "type": "view", "dwell_ms": 7}])
    conn = db.connect()
    signals = main.recent_signals(conn, uid)
    conn.close()
    assert len(signals) == main.MAX_SIGNALS
    assert [(s.card_id, s.kind, s.dwell_ms) for s in signals[-3:]] == [
        (2, "like", None), (3, "unlike", None), (4, "view", 7)]     # oldest first


def test_quiz_only_looks_at_newest_300_views(client):
    start_session(client, ["tech"])
    uid = client.cookies.get("rr_uid")
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 9000}])
    insert_views(uid, 1, main.QUIZ_LOOKBACK_VIEWS)      # 300 newer views push 301 out
    assert quizzes(feed(client)) == []


def test_quiz_still_found_within_newest_300_views(client):
    start_session(client, ["tech"])
    uid = client.cookies.get("rr_uid")
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 9000}])
    insert_views(uid, 1, main.QUIZ_LOOKBACK_VIEWS - 1)
    assert quizzes(feed(client))[0]["quiz"]["card_id"] == 301


# 3. card_id must fit in SQLite's signed 64-bit integer.

def test_huge_card_id_is_422_not_500(client):
    uid = start_session(client, ["tech"])
    for bad in (2**63, 2**70, -1):
        assert send_events(client, [{"card_id": bad, "type": "like"}]).status_code == 422
        assert client.post("/api/quiz/answer",
                           json={"card_id": bad, "choice": 0}).status_code == 422
    # The largest allowed id is fine: unknown card -> dropped / 404.
    assert send_events(client, [{"card_id": 2**63 - 1, "type": "like"}]).status_code == 204
    assert client.post("/api/quiz/answer",
                       json={"card_id": 2**63 - 1, "choice": 0}).status_code == 404
    assert query("SELECT * FROM events WHERE user_id = ?", (uid,)) == []


# 4. Only quizzes that were shown can be answered.

def test_cannot_answer_quiz_never_shown(client):
    uid = start_session(client, ["tech"])
    r = client.post("/api/quiz/answer", json={"card_id": 301, "choice": 2})
    assert r.status_code == 404
    assert "answer_index" not in r.text
    assert quiz_events(uid) == []


def test_quiz_shown_to_another_user_does_not_count(client):
    other = start_session(client, ["tech"])
    mark_quiz_shown(other, 301)
    client.cookies.clear()
    start_session(client, ["tech"])
    assert client.post("/api/quiz/answer", json={"card_id": 301, "choice": 2}).status_code == 404


def test_quiz_answerable_after_feed_shows_it(client):
    start_session(client, ["tech"])
    send_events(client, [{"card_id": 302, "type": "view", "dwell_ms": 6000}])
    assert quizzes(feed(client))[0]["quiz"]["card_id"] == 302
    r = client.post("/api/quiz/answer", json={"card_id": 302, "choice": 1})
    assert r.status_code == 200 and r.json() == {"correct": True, "answer_index": 1}


# 5. Cross-site requests to /api are refused, GETs included.

def test_cross_site_get_feed_403_and_no_state_change(client):
    uid = start_session(client, ["tech"])
    send_events(client, [{"card_id": 301, "type": "view", "dwell_ms": 6000}])
    r = client.get("/api/feed", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert_security_headers(r)
    assert query("SELECT * FROM served WHERE user_id = ?", (uid,)) == []
    assert query("SELECT * FROM events WHERE user_id = ? AND type = 'quiz_shown'", (uid,)) == []
    for path in ("/api/me", "/api/topics"):
        assert client.get(path, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert send_events(client, []).status_code == 204
    r = client.post("/api/events", json={"events": []}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


@pytest.mark.parametrize("value", ["same-origin", "same-site", "none", None])
def test_non_cross_site_fetch_allowed(client, value):
    start_session(client, ["tech"])
    headers = {"Sec-Fetch-Site": value} if value else {}
    assert client.get("/api/feed", headers=headers).status_code == 200


def test_cross_site_static_files_still_served(client):
    # Only /api is protected; a cross-site link to the app itself must work.
    r = client.get("/", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code != 403
