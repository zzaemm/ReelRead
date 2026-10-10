# ReelRead V1 — Build Contract

This is the contract every part of V1 builds against. If code and this file disagree, this file wins; change it deliberately.

## What V1 is

A mobile-first web app (PWA). On first open you pick 1–5 topics. Then you get a full-screen vertical swipe feed of Wikipedia summary cards. The feed learns from how long you linger on cards and what you like. About every 8th item is a recall question about a card you lingered on earlier.

Out of scope for V1: accounts/login, news/RSS, social features, LLMs, deployment.

## Layout and ownership

```
reelread/
  SPEC.md, DECISIONS.md, README.md, requirements.txt, .gitignore
  backend/
    db.py            schema + connect()                         [overseer — do not change schema without asking]
    fetch_cards.py   Wikipedia -> cards                         [overseer]
    ranker.py        user vector + ranking                      [engine builder]
    quiz.py          rule-based recall questions                [engine builder]
    build_index.py   offline: vectors file + quizzes table      [engine builder]
    main.py          FastAPI app, all /api routes, static files [backend builder]
    security.py      cookie, rate limiting, headers             [backend builder]
    tests/           pytest; test_<module>.py per owner
  frontend/          index.html, style.css, app.js, sw.js,
                     manifest.webmanifest, icons/               [frontend builder]
```

Python 3.10+. Dependencies are limited to: fastapi, uvicorn, requests, numpy, scikit-learn, pytest, httpx. Frontend is plain HTML/CSS/JS: no frameworks, no CDNs, no build step.

## Topics

Topic ids, from `db.TOPICS`: `tech`, `sports`, `ai_ml`, `finance`, `politics`. Labels: Tech, Sports, AI & ML, Finance, Politics.

## Database

The schema is in `backend/db.py`. Tables: `cards`, `card_topics`, `users`, `events`, `served`, `quizzes`. `DB_PATH` can be overridden with the env var `REELREAD_DB`. Tests must use a temp DB.

## Engine interface (ranker.py / quiz.py / build_index.py)

`python build_index.py` runs offline after fetch. It:

1. Fits TF-IDF on `title + ". " + extract` for all cards (`stop_words="english"`, `sublinear_tf=True`, `min_df=2`, `max_df=0.5`), reduces with TruncatedSVD to 128 dims (or fewer if there are not enough cards), and L2-normalises. It saves `backend/index.npz` (override: `--index PATH` CLI flag, or env `REELREAD_INDEX`) with arrays `card_ids` (int64) and `vectors` (float32, N×D). `random_state` is fixed.
2. Rebuilds the `quizzes` table: delete all rows, then insert one row per card where a good question can be made.

`ranker.py` exposes:

```python
class Ranker:
    @classmethod
    def load(cls, path) -> "Ranker"        # missing file -> Ranker with no vectors (everything random)
    def rank(self, signals: list[Signal], candidate_ids: list[int], k: int,
             explore_frac: float = 0.2, rng: random.Random | None = None) -> list[int]
    def neighbours(self, card_id: int, n: int) -> list[int]   # nearest cards by cosine, excluding itself

@dataclass
class Signal:
    card_id: int
    kind: str          # "view" | "like" | "unlike"
    dwell_ms: int | None
```

Signal weights for building the user vector, as a sum of weight × card vector:

- view: `min(dwell_ms / 10000, 1.0)`.
- view with `dwell_ms < 1500`: `-0.5`. This is a skip.
- like: `+1.0`.
- unlike: cancels one like (`-1.0`).

`rank()` returns up to k distinct ids, all drawn from `candidate_ids`. If the user vector is zero or the index is empty, return a random sample. Otherwise, `round(k * (1 - explore_frac))` ids are the top cosine matches, and the rest are random picks from the remaining candidates. The two groups are interleaved, not just appended. Candidates without a vector count as score 0. Rank must take under 50 ms for 5,000 candidates.

`quiz.py` exposes `make_quiz(card, distractor_titles: list[str], rng) -> dict | None`. It returns `{"question": str, "options": [4 titles], "answer_index": int}` or None. The rules:

- Take the first sentence of the extract.
- Remove parenthetical asides such as pronunciations and dates.
- Replace the subject with `___`, case-insensitive. The subject is the title, or the title with its trailing "(disambiguator)" removed, e.g. "Python (programming language)" becomes "Python".
- If the subject isn't found, or the question would be under 40 chars, return None.
- If the question still contains the answer string anywhere, return None.
- Distractors are 3 other card titles. `build_index.py` prefers the 6 nearest same-topic neighbours (from the top 40) and picks 3 of them at random, falling back to any top-40 neighbour, then to random topic-mates. None can equal, contain or be contained in the answer, and all are distinct. Shuffle the options with rng. (Changed after a real-data check: a random 5–40 band let in off-topic options like "Hawaii" for "Nuclear weapon".)

## HTTP API (main.py)

Everything is same-origin. FastAPI serves `frontend/` at `/` (index.html at `/`). All JSON. Errors use FastAPI's `{"detail": ...}` with a correct status code.

User identity: cookie `rr_uid` holds a UUID4. It's set with `HttpOnly`, `SameSite=Lax`, `Path=/`, and `Max-Age` of 1 year. `Secure` is set only when the env `REELREAD_SECURE_COOKIE=1`, because the local dev server runs over plain http. A missing, malformed, or unknown cookie gets `401` on routes that need a user.

| Method & path | Request | Response |
|---|---|---|
| `GET /api/topics` | – | `200 [{"id":"tech","label":"Tech","count":412}, ...]`, where count is the number of cards in that topic |
| `POST /api/session` | `{"topics":["tech","ai_ml"]}`, 1–5 unique valid ids | `200 {"topics":[...]}` and sets the cookie. Valid existing cookie: update that user's topics. Otherwise: create a new user. Invalid topics: `422` |
| `GET /api/me` | cookie | `200 {"topics":[...]}`, or `401` |
| `GET /api/feed?n=10` | cookie; n from 1 to 20, default 10 | `200 {"items":[Item, ...]}` (see below) |
| `POST /api/events` | `{"events":[{"card_id":1,"type":"view","dwell_ms":5300},{"card_id":2,"type":"like"}]}` | `204` |
| `POST /api/quiz/answer` | `{"card_id":1,"choice":2}` | `200 {"correct":true,"answer_index":2}` |

An `Item` is one of:

```json
{"kind":"card","card":{"id":1,"title":"...","extract":"...","url":"https://en.wikipedia.org/wiki/...","thumbnail":"https://thumb.wikimedia.org/..." | null,"topics":["tech"],"liked":false}}
{"kind":"quiz","quiz":{"card_id":1,"question":"___ is a ...","options":["A","B","C","D"]}}
```

The quiz item never includes `answer_index`. The client learns the answer only by answering.

**Feed rules:**

- Candidates are cards in the user's topics that are not in `served` for that user.
- If fewer than n are left, clear that user's `served` rows and start over.
- Signals are the user's last 300 view/like/unlike events.
- Use `Ranker.rank`, then insert the chosen ids into `served`.
- Quiz insertion: in each feed response, add at most 1 quiz, placed after the 8th card, or at the end if there are fewer than 8 cards. The quiz is about a card that meets all of these:
  - the user viewed it with a total dwell of at least 5000 ms;
  - it has a `quizzes` row;
  - it has never had a quiz_shown, quiz_correct or quiz_wrong event from this user (so a skipped quiz is not repeated);
  - it isn't one of the cards in this same response.
  
  Pick the most recently viewed eligible card and log a server-side `quiz_shown` event for it. If none is eligible, there's no quiz.
- `liked` is true when the count of the user's likes exceeds their unlikes for that card.

**Events rules:**

- Accept at most 50 events per request. Allowed types: `view`, `like`, `unlike`. Clients cannot send `quiz_*`.
- The `card_id` must exist. Unknown ids are silently dropped, not an error.
- `dwell_ms` is required for view and is clamped to 0–600000.
- Body larger than 16 KB: `413`.

**Quiz answer rules:**

- `choice` is 0–3.
- The card must have a quiz that was shown to this user (a `quiz_shown` event exists), otherwise `404`. This stops a client pulling the answer key for quizzes it never saw.
- If the user already answered this card's quiz, return the result again without logging another event.
- Otherwise, log quiz_correct or quiz_wrong.

**Request size:** every request body is capped (16 KB for `/api/events`, 4 KB elsewhere) before parsing; over the cap → `413`. Validation errors (`422`) never echo the submitted input back.

**Storage caps:** each user keeps at most their 2,000 most recent view/like/unlike events; older ones are deleted on insert. Quiz eligibility looks only at the user's 300 most recent view events.

**Rate limiting:** in-memory, per user id, with per-IP limiting for session creation.

- `/api/events`: 60 requests/min.
- `/api/feed`: 30/min.
- `/api/quiz/answer`: 30/min.
- `POST /api/session` creating a new user: 10/min per client IP.

Over the limit: `429`.

**Security headers on every response:**

- `Content-Security-Policy: default-src 'self'; img-src 'self' https://upload.wikimedia.org https://thumb.wikimedia.org data:; script-src 'self'; style-src 'self'; connect-src 'self'; manifest-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `X-Frame-Options: DENY`

**CSRF:** state-changing routes (all POSTs) must reject requests whose `Origin` header is present and doesn't match the request's own host. Missing Origin is allowed, because sendBeacon and same-origin requests can omit it. Every `/api/*` request (GET included, since `/api/feed` writes `served` and `quiz_shown`) with `Sec-Fetch-Site: cross-site` gets `403`.

All integer ids in request bodies are bounded to 0–2^63−1 (SQLite's integer range).

On startup, main.py calls `init_db()` and loads `Ranker.load(path)`, where path is env `REELREAD_INDEX` or `backend/index.npz` by default. Run with `uvicorn main:app --host 0.0.0.0 --port 8000` from `backend/`.

## Frontend rules

- Never put server data into `innerHTML`. Use `textContent` and `createElement` only. Card text is third-party content.
- Before using URLs: `img.src` only if the thumbnail starts with `https://upload.wikimedia.org/` or `https://thumb.wikimedia.org/`, and `a.href` only if the URL starts with `https://en.wikipedia.org/`. Outbound links use `target="_blank" rel="noopener noreferrer"`.
- No inline `<script>`, no inline `style=""` attributes, and no inline event handlers. The CSP forbids them.
- Dwell tracking: an IntersectionObserver with a 0.6 threshold times each card while visible. The timer pauses when the page is hidden. Queue `view` events and flush them every 10 s, after 10 queued events, and on `visibilitychange`→hidden / `pagehide` via `navigator.sendBeacon('/api/events', new Blob([json], {type:'application/json'}))`.
- Prefetch the next page when the user is 3 items from the end. Never show the same card twice in one session.
- Service worker: cache-first for the app shell (html/css/js/manifest/icons), and network-only for `/api/*`, which is never cached.
