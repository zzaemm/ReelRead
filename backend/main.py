"""ReelRead web server: the JSON API under /api plus the static frontend at /.

Run from backend/:   uvicorn main:app --host 0.0.0.0 --port 8000

How a request flows through this file:
  1. Middleware (security.py): add security headers, reject cross-site
     requests (CSRF), and cap the request body size (413).
  2. Route function: dependencies first resolve the user from the cookie (401
     if missing/unknown) and apply the rate limit (429), then the body is
     validated by a Pydantic model (422 on bad input).
  3. The route opens a short-lived SQLite connection, runs parameterised SQL
     (always `?` placeholders, never string formatting of user data), and
     returns JSON.

SPEC.md is the contract for every route here.
"""
import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

import db
import security
from ranker import Ranker, Signal

log = logging.getLogger("reelread")

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# Feed / quiz tuning, straight from SPEC.md.
MAX_SIGNALS = 300            # how many recent view/like/unlike events feed the ranker
QUIZ_AFTER_N_CARDS = 8       # the quiz goes after the 8th card
QUIZ_MIN_DWELL_MS = 5000     # total view time on a card before we quiz on it
MAX_DWELL_MS = 600_000       # 10 minutes; longer "views" are clamped
MAX_EVENTS_PER_REQUEST = 50
MAX_STORED_SIGNALS = 2000    # per user we keep only the newest 2,000 view/like/unlike events
QUIZ_LOOKBACK_VIEWS = 300    # quiz eligibility only looks at the newest 300 views
MAX_SQLITE_INT = 2**63 - 1   # SQLite integers are signed 64-bit; bigger ids would crash
# Wikimedia serves thumbnails from either CDN host; anything else is dropped.
THUMBNAIL_PREFIXES = ("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")


def index_path() -> Path:
    """Where build_index.py wrote the card vectors (env REELREAD_INDEX overrides)."""
    return Path(os.environ.get("REELREAD_INDEX", BACKEND_DIR / "index.npz"))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: create tables, load the ranking index."""
    db.init_db()
    path = index_path()
    app.state.ranker = Ranker.load(path)
    log.info("ReelRead started (db=%s, index=%s, exists=%s)", db.DB_PATH, path, path.exists())
    yield


app = FastAPI(title="ReelRead", lifespan=lifespan,
              # No public API docs: this is an app backend, not a public API.
              docs_url=None, redoc_url=None, openapi_url=None)

# Starlette runs the LAST added middleware FIRST, so requests pass through:
#   SecurityHeaders -> OriginCheck -> BodySizeLimit -> routes.
# The headers middleware is outermost, so even the 403/413 rejections get
# security headers, and cross-site requests are refused before we read a body.
app.add_middleware(security.BodySizeLimitMiddleware)
app.add_middleware(security.OriginCheckMiddleware)
app.add_middleware(security.SecurityHeadersMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 for bad input, WITHOUT echoing the submitted data back.

    FastAPI's default handler includes each error's "input" (and "ctx"), which
    reflects whatever the client sent. We keep only where it failed ("loc"),
    what kind of error ("type") and the message ("msg").
    """
    errors = [{k: v for k, v in err.items() if k in ("type", "loc", "msg")}
              for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def get_ranker(request: Request) -> Ranker:
    """The Ranker loaded at startup (loaded lazily if startup didn't run)."""
    ranker = getattr(request.app.state, "ranker", None)
    if ranker is None:
        ranker = request.app.state.ranker = Ranker.load(index_path())
    return ranker


@contextmanager
def open_db():
    """A short-lived connection per request: commit on success, roll back on
    error, and always close (so we never leak file handles)."""
    conn = db.connect()
    try:
        with conn:          # sqlite3: commits, or rolls back on exception
            yield conn
    finally:
        conn.close()


def placeholders(values) -> str:
    """'?,?,?' for an SQL IN (...) clause. Only the NUMBER of placeholders comes
    from the data; the values themselves are still passed as parameters."""
    return ",".join("?" * len(values))


# ---------------------------------------------------------------------------
# Users (cookie -> user) and rate-limit dependencies
# ---------------------------------------------------------------------------

@dataclass
class User:
    id: str
    topics: List[str]


def load_user(user_id: Optional[str]) -> Optional[User]:
    """Look the cookie value up in the users table; None if it's bad or unknown."""
    if not security.is_valid_uuid4(user_id):
        return None
    with open_db() as conn:
        row = conn.execute("SELECT id, topics FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return User(id=row["id"], topics=json.loads(row["topics"]))


def current_user(request: Request) -> User:
    """Dependency for routes that need a user: 401 for a missing, malformed or unknown cookie."""
    user = load_user(request.cookies.get(security.COOKIE_NAME))
    if user is None:
        raise HTTPException(status_code=401, detail="No session. POST /api/session first.")
    return user


def rate_limited_user(limiter: security.RateLimiter):
    """Build a dependency that resolves the user (401 first) and then applies
    `limiter` keyed by that user's id (429)."""
    def dependency(user: User = Depends(current_user)) -> User:
        security.enforce_rate_limit(limiter, user.id)
        return user
    return dependency


# ---------------------------------------------------------------------------
# Request bodies (Pydantic validates these; bad input -> 422 automatically)
# ---------------------------------------------------------------------------

class SessionIn(BaseModel):
    topics: List[str] = Field(min_length=1, max_length=5)

    @field_validator("topics")
    @classmethod
    def topics_must_be_known_and_unique(cls, topics: List[str]) -> List[str]:
        # The message deliberately doesn't repeat the bad values (no echo).
        if any(t not in db.TOPICS for t in topics):
            raise ValueError("unknown topic id")
        if len(set(topics)) != len(topics):
            raise ValueError("topics must be unique")
        return topics


class EventIn(BaseModel):
    card_id: StrictInt = Field(ge=0, le=MAX_SQLITE_INT)
    # quiz_* events are deliberately NOT allowed here: only the server logs them,
    # otherwise a client could mark quizzes as answered without answering.
    type: Literal["view", "like", "unlike"]
    # A float is accepted because browser timers (performance.now()) produce
    # fractional milliseconds; we round it to an int below.
    dwell_ms: Optional[float] = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_dwell(self) -> "EventIn":
        if self.type == "view":
            if self.dwell_ms is None:
                raise ValueError("dwell_ms is required for view events")
            # Clamp to 0..10 min so one bogus value can't dominate the ranking.
            self.dwell_ms = int(min(max(round(self.dwell_ms), 0), MAX_DWELL_MS))
        else:
            self.dwell_ms = None  # only views carry a dwell time
        return self


class EventsIn(BaseModel):
    events: List[EventIn] = Field(max_length=MAX_EVENTS_PER_REQUEST)


class QuizAnswerIn(BaseModel):
    card_id: StrictInt = Field(ge=0, le=MAX_SQLITE_INT)
    choice: StrictInt = Field(ge=0, le=3)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/topics")
def get_topics():
    """All topics with how many cards each one has."""
    with open_db() as conn:
        rows = conn.execute(
            "SELECT topic, COUNT(*) AS n FROM card_topics GROUP BY topic"
        ).fetchall()
    counts = {row["topic"]: row["n"] for row in rows}
    return [{"id": tid, "label": label, "count": counts.get(tid, 0)}
            for tid, label in db.TOPICS.items()]


@app.post("/api/session")
def post_session(body: SessionIn, request: Request, response: Response):
    """Create an anonymous user, or update the topics of the current one."""
    topics_json = json.dumps(body.topics)
    existing = load_user(request.cookies.get(security.COOKIE_NAME))

    if existing is not None:
        with open_db() as conn:
            conn.execute("UPDATE users SET topics = ? WHERE id = ?", (topics_json, existing.id))
        user_id = existing.id
    else:
        # Creating users is limited per IP so a script can't fill the DB.
        security.enforce_rate_limit(security.session_create_limiter, security.client_ip(request))
        user_id = security.new_user_id()
        with open_db() as conn:
            conn.execute("INSERT INTO users (id, topics) VALUES (?, ?)", (user_id, topics_json))

    # (Re)set the cookie either way, which also refreshes its 1-year expiry.
    security.set_user_cookie(response, user_id)
    return {"topics": body.topics}


@app.get("/api/me")
def get_me(user: User = Depends(current_user)):
    return {"topics": user.topics}


@app.get("/api/feed")
def get_feed(
    request: Request,
    n: int = Query(10, ge=1, le=20),
    user: User = Depends(rate_limited_user(security.feed_limiter)),
):
    """The next n cards for this user, plus at most one recall quiz.

    Follows "Feed rules" in SPEC.md step by step.
    """
    ranker = get_ranker(request)
    with open_db() as conn:
        # 1. Candidates: cards in the user's topics not yet served to them.
        candidates = unserved_candidates(conn, user)
        # 2. Running out? Forget what we've served and start over.
        if len(candidates) < n:
            conn.execute("DELETE FROM served WHERE user_id = ?", (user.id,))
            candidates = unserved_candidates(conn, user)

        # 3. Signals: the last 300 view/like/unlike events (oldest first).
        signals = recent_signals(conn, user.id)

        # 4. Rank. The ranker is a separate module, so we double-check its
        #    output: only real candidates, no duplicates, at most n.
        ranked = ranker.rank(signals, candidates, n)
        candidate_set = set(candidates)
        chosen = [cid for cid in dict.fromkeys(ranked) if cid in candidate_set][:n]

        # 5. Remember what we served so the next page doesn't repeat it.
        conn.executemany(
            "INSERT OR IGNORE INTO served (user_id, card_id) VALUES (?, ?)",
            [(user.id, cid) for cid in chosen],
        )

        items = [{"kind": "card", "card": card} for card in load_cards(conn, user.id, chosen)]

        # 6. At most one quiz: after the 8th card, or at the end if fewer.
        quiz = pick_quiz(conn, user.id, exclude=chosen)
        if quiz is not None:
            items.insert(min(QUIZ_AFTER_N_CARDS, len(items)), {"kind": "quiz", "quiz": quiz})

    return {"items": items}


@app.post("/api/events", status_code=204)
async def post_events(
    request: Request,
    user: User = Depends(rate_limited_user(security.events_limiter)),
):
    """Store view/like/unlike events. Returns 204 No Content.

    The body (already capped at 16 KB by BodySizeLimitMiddleware) is parsed
    here rather than via a Pydantic parameter so that it's accepted whatever
    Content-Type the client sends: navigator.sendBeacon can't always set one.
    """
    raw = await request.body()
    try:
        payload = EventsIn.model_validate_json(raw)
    except ValidationError as exc:
        # Re-raise as FastAPI's own error so the client gets the usual 422 shape.
        errors = exc.errors(include_url=False, include_context=False)
        raise RequestValidationError([{**e, "loc": ("body", *e["loc"])} for e in errors])

    # SQLite calls block, so run them on the threadpool, not the event loop.
    await run_in_threadpool(store_events, user.id, payload.events)
    return Response(status_code=204)


@app.post("/api/quiz/answer")
def post_quiz_answer(
    body: QuizAnswerIn,
    user: User = Depends(rate_limited_user(security.quiz_limiter)),
):
    """Check a quiz answer. Answering twice returns the first result again."""
    with open_db() as conn:
        # BEGIN IMMEDIATE takes the write lock now, so two simultaneous answers
        # can't both see "not answered yet" and both log an event.
        conn.execute("BEGIN IMMEDIATE")
        quiz = conn.execute(
            "SELECT answer_index FROM quizzes WHERE card_id = ?", (body.card_id,)
        ).fetchone()
        # Only quizzes the feed actually showed this user can be answered,
        # otherwise a client could pull the whole answer key card by card.
        shown = conn.execute(
            "SELECT 1 FROM events WHERE user_id = ? AND card_id = ? AND type = 'quiz_shown' LIMIT 1",
            (user.id, body.card_id),
        ).fetchone()
        if quiz is None or shown is None:
            raise HTTPException(status_code=404, detail="No quiz for this card.")
        answer_index = quiz["answer_index"]

        previous = conn.execute(
            "SELECT type FROM events WHERE user_id = ? AND card_id = ? "
            "AND type IN ('quiz_correct', 'quiz_wrong') ORDER BY id LIMIT 1",
            (user.id, body.card_id),
        ).fetchone()
        if previous is not None:
            return {"correct": previous["type"] == "quiz_correct", "answer_index": answer_index}

        correct = body.choice == answer_index
        conn.execute(
            "INSERT INTO events (user_id, card_id, type) VALUES (?, ?, ?)",
            (user.id, body.card_id, "quiz_correct" if correct else "quiz_wrong"),
        )
    return {"correct": correct, "answer_index": answer_index}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
               include_in_schema=False)
def api_not_found(path: str):
    """Unknown /api/... paths get a JSON 404 instead of falling through to the static files."""
    raise HTTPException(status_code=404, detail="Not found.")


# ---------------------------------------------------------------------------
# Feed / events helpers (plain functions, each takes an open connection)
# ---------------------------------------------------------------------------

def unserved_candidates(conn, user: User) -> List[int]:
    """Ids of cards in the user's topics that haven't been served to them yet."""
    if not user.topics:
        return []
    rows = conn.execute(
        f"SELECT DISTINCT card_id FROM card_topics "
        f"WHERE topic IN ({placeholders(user.topics)}) "
        f"AND card_id NOT IN (SELECT card_id FROM served WHERE user_id = ?) "
        f"ORDER BY card_id",
        (*user.topics, user.id),
    ).fetchall()
    return [row["card_id"] for row in rows]


# A user's view/like/unlike events, newest first.
#
# Why three SELECTs glued with UNION ALL instead of `type IN ('view','like','unlike')`?
# With the index events(user_id, type, id), each SELECT below reads one
# (user, type) slice of the index that is already sorted by id. SQLite can then
# MERGE the three sorted slices and stop after LIMIT rows. With `type IN (...)`
# it would instead read ALL of the user's events and sort them in a temporary
# B-tree before applying LIMIT, which got slow with a huge event history.
# (Check with: EXPLAIN QUERY PLAN <this query>; there should be no "TEMP B-TREE".)
SIGNAL_EVENTS_NEWEST_FIRST = """
    SELECT id, card_id, type, dwell_ms FROM events WHERE user_id = :uid AND type = 'view'
    UNION ALL
    SELECT id, card_id, type, dwell_ms FROM events WHERE user_id = :uid AND type = 'like'
    UNION ALL
    SELECT id, card_id, type, dwell_ms FROM events WHERE user_id = :uid AND type = 'unlike'
    ORDER BY id DESC
"""
RECENT_SIGNALS_SQL = SIGNAL_EVENTS_NEWEST_FIRST + " LIMIT :limit"


def recent_signals(conn, user_id: str) -> List[Signal]:
    """The user's last MAX_SIGNALS view/like/unlike events as ranker Signals,
    oldest first. quiz_* events are not ranking signals."""
    rows = conn.execute(RECENT_SIGNALS_SQL, {"uid": user_id, "limit": MAX_SIGNALS}).fetchall()
    return [Signal(card_id=r["card_id"], kind=r["type"], dwell_ms=r["dwell_ms"])
            for r in reversed(rows)]


def load_cards(conn, user_id: str, card_ids: List[int]) -> List[dict]:
    """Card JSON objects (with `topics` and `liked`) in the same order as card_ids."""
    if not card_ids:
        return []
    marks = placeholders(card_ids)
    rows = conn.execute(
        f"SELECT id, title, extract, url, thumbnail FROM cards WHERE id IN ({marks})",
        card_ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}

    topics: dict = {}
    for row in conn.execute(
        f"SELECT card_id, topic FROM card_topics WHERE card_id IN ({marks}) ORDER BY topic",
        card_ids,
    ):
        topics.setdefault(row["card_id"], []).append(row["topic"])

    # liked = more likes than unlikes for this card.
    liked = {
        row["card_id"]: row["balance"] > 0
        for row in conn.execute(
            f"SELECT card_id, SUM(type = 'like') - SUM(type = 'unlike') AS balance "
            f"FROM events WHERE user_id = ? AND card_id IN ({marks}) "
            f"AND type IN ('like', 'unlike') GROUP BY card_id",
            (user_id, *card_ids),
        )
    }

    cards = []
    for cid in card_ids:
        row = by_id.get(cid)
        if row is None:
            continue
        thumb = row["thumbnail"]
        cards.append({
            "id": row["id"],
            "title": row["title"],
            "extract": row["extract"],
            "url": row["url"],
            # Defence in depth: the frontend checks this too.
            "thumbnail": thumb if thumb and thumb.startswith(THUMBNAIL_PREFIXES) else None,
            "topics": topics.get(cid, []),
            "liked": liked.get(cid, False),
        })
    return cards


def pick_quiz(conn, user_id: str, exclude: List[int]) -> Optional[dict]:
    """The quiz for the most recently viewed eligible card, or None.

    Eligible (SPEC.md): total view dwell >= 5000 ms, has a quizzes row, never
    shown to or answered by this user, and not one of the cards in this response.
    Only the user's newest QUIZ_LOOKBACK_VIEWS view events are considered, so
    the cost doesn't grow with their whole history.
    Picking a quiz logs a 'quiz_shown' event, so a skipped quiz never comes back.
    The answer_index is never selected, so it can't leak to the client.
    """
    exclude_sql = f"AND v.card_id NOT IN ({placeholders(exclude)})" if exclude else ""
    row = conn.execute(
        f"""
        WITH recent_views AS (           -- the user's newest views only
            SELECT id, card_id, dwell_ms FROM events
            WHERE user_id = ? AND type = 'view'
            ORDER BY id DESC LIMIT ?
        )
        SELECT q.card_id, q.question, q.options
        FROM recent_views v
        JOIN quizzes q ON q.card_id = v.card_id
        WHERE v.card_id NOT IN (
              SELECT card_id FROM events
              WHERE user_id = ? AND type IN ('quiz_shown', 'quiz_correct', 'quiz_wrong'))
          {exclude_sql}
        GROUP BY q.card_id
        HAVING SUM(v.dwell_ms) >= ?
        ORDER BY MAX(v.id) DESC          -- most recently viewed first
        LIMIT 1
        """,
        (user_id, QUIZ_LOOKBACK_VIEWS, user_id, *exclude, QUIZ_MIN_DWELL_MS),
    ).fetchone()
    if row is None:
        return None
    conn.execute("INSERT INTO events (user_id, card_id, type) VALUES (?, ?, 'quiz_shown')",
                 (user_id, row["card_id"]))
    return {"card_id": row["card_id"], "question": row["question"],
            "options": json.loads(row["options"])}


def store_events(user_id: str, events: List[EventIn]) -> None:
    """Insert events, silently dropping ones whose card_id doesn't exist, then
    trim the user's view/like/unlike history to the newest MAX_STORED_SIGNALS.

    The trim keeps the events table (and every query on it) from growing
    without bound. quiz_* events are never deleted: they record which quizzes
    were shown/answered, and losing them would let old quizzes come back.
    """
    if not events:
        return
    with open_db() as conn:
        ids = list({e.card_id for e in events})
        known = {row["id"] for row in conn.execute(
            f"SELECT id FROM cards WHERE id IN ({placeholders(ids)})", ids)}
        conn.executemany(
            "INSERT INTO events (user_id, card_id, type, dwell_ms) VALUES (?, ?, ?, ?)",
            [(user_id, e.card_id, e.type, e.dwell_ms) for e in events if e.card_id in known],
        )
        # id of the MAX_STORED_SIGNALS-th newest signal event (None if fewer).
        cutoff = conn.execute(
            SIGNAL_EVENTS_NEWEST_FIRST + " LIMIT 1 OFFSET :skip",
            {"uid": user_id, "skip": MAX_STORED_SIGNALS - 1},
        ).fetchone()
        if cutoff is not None:
            conn.execute(
                "DELETE FROM events WHERE user_id = ? AND type IN ('view', 'like', 'unlike') "
                "AND id < ?",
                (user_id, cutoff["id"]),
            )


# ---------------------------------------------------------------------------
# Static frontend. Mounted LAST so every /api route above takes priority.
# ---------------------------------------------------------------------------

if FRONTEND_DIR.is_dir():
    # html=True serves index.html for "/". check_dir=False is harmless here and
    # keeps startup working even while the frontend folder is still empty.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True, check_dir=False),
              name="frontend")
else:
    log.warning("Frontend folder %s not found; serving the API only.", FRONTEND_DIR)
