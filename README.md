# ReelRead

Reels, but for reading. A full-screen vertical swipe feed of short Wikipedia reads on the topics you pick. It learns from how long you linger on each card, and every so often it quizzes you on something you read a few cards back.

V1 is a local web app (PWA): a FastAPI + SQLite backend and a plain HTML/CSS/JS frontend. `SPEC.md` is the contract every part was built against, and `DECISIONS.md` records why things are the way they are.

## Run it (Windows, PowerShell)

You need Python 3.10 or newer (`python --version`).

```powershell
cd $HOME\Documents\reelread
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

The database (`reelread.db`, 1,845 real cards) and `index.npz` are included, so you don't need to download anything first.

**On your phone:** connect to the same Wi-Fi, then open `http://<your-PC's-IP>:8000`. To find the IP, run `ipconfig` and look for the IPv4 address. Allow Python through Windows Firewall when it asks. Everything works over plain http, except "install to home screen" and offline caching: browsers only allow service workers on https or localhost. That's a V2 (deployment) problem.

## Change the content

```powershell
cd backend
# 1. edit seeds\*.txt: delete any line you don't want as a card
python fetch_cards.py fetch     # syncs the DB to the seed files; only new lines hit the network
python build_index.py           # rebuilds vectors + quizzes (about 5 s)
# restart uvicorn so it loads the new index
```

To start from scratch, run `python fetch_cards.py collect`. It regenerates the seed lists from Wikipedia's "Outline of …" pages, using a fixed random sample of 400 titles per topic. Edit `TOPICS` in `fetch_cards.py` to change which outline pages each topic uses. Wikimedia rate-limits shared networks heavily, so expect pauses of up to a minute.

## How it works

```
fetch_cards.py ──> reelread.db (cards, card_topics)
build_index.py ──> index.npz (one 128-d vector per card)  +  quizzes table
main.py (FastAPI) serves /api/* and frontend/
   GET  /api/feed     rank unseen cards in your topics, insert ≤1 quiz
   POST /api/events   view (with dwell time), like, unlike
```

**Ranking (`ranker.py`).** Each card is a TF-IDF vector of its words, compressed to 128 dimensions with SVD. Your taste is a weighted sum of the vectors of cards you've interacted with:

| Signal | Weight |
|---|---|
| Reading time | up to +1 at 10 s |
| A skip (under 1.5 s) | −0.5 |
| A like | +1 |

Unseen cards are ranked by cosine similarity to your taste vector. 80% of each page is the best matches and 20% is random, so the feed can still surprise you.

**Quizzes (`quiz.py`).** Each quiz takes a card's first sentence with its subject blanked out, plus 4 titles to choose from. The 3 wrong options are the nearest same-topic articles, so they're plausible. A quiz only appears for a card you spent at least 5 s on, and it appears at most once.

## Security model

Assume every Wikipedia title and extract is attacker-controlled, because anyone can edit Wikipedia.

- **Ingest.**
  - Only plain-text extracts are stored, never HTML.
  - Thumbnails are kept only from Wikimedia's CDNs, and links only to en.wikipedia.org.
- **Browser.**
  - All server data goes in via `textContent`, never `innerHTML`.
  - URLs are checked against the same prefixes again on the client.
  - A strict CSP (`script-src 'self'`, no inline code), plus `X-Frame-Options: DENY` and `nosniff`.
- **Identity.** An anonymous UUID4 in an `HttpOnly`, `SameSite=Lax` cookie. Set `REELREAD_SECURE_COOKIE=1` once it's served over https.
- **API abuse.**
  - Pydantic validation, and parameterised SQL everywhere.
  - Per-user rate limits; new users are limited per IP.
  - Request bodies are size-capped, and each user's event history is capped.
  - Clients can't log quiz results themselves, and the server never sends answers before you answer.
  - You can only answer quizzes you were shown.
- **CSRF.** POSTs with a foreign `Origin` are rejected, and so is any `/api` request marked `Sec-Fetch-Site: cross-site`.

Known V1 limits: rate limits are in memory (one server process only), and there's no auth, so anyone with your cookie is "you".

## Tests

```powershell
cd backend
python -m pytest tests -q        # 130 tests: API, security, ranker, quiz
```

## Layout

```
backend/   main.py (API) · security.py · ranker.py · quiz.py · build_index.py
           fetch_cards.py · db.py (schema) · seeds/ · tests/
frontend/  index.html · style.css · app.js · sw.js · manifest.webmanifest · icons/
```
