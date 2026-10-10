# Decisions

1. Platform: Web app (PWA). Fastest to test the idea, uses HTML/JS I already know. Revisit native if people use it.
2. Content: Wikipedia only for V1. Free, legal, evergreen. News (RSS) planned for V2.
3. Backend: FastAPI + SQLite. Python matches the ML ranking work.
4. Topics: Tech, Sports, AI & ML, Finance, Politics.
5. AI and ML merged into one topic ("AI & ML"). Heavy overlap on Wikipedia; splitting is curation work that doesn't test the core idea.
6. Seeding: Wikipedia "Outline of X" pages, then manually trim off-topic articles. Good quality for minimal effort.

## Proposed (taken as defaults during the V1 build — confirm or change)

7. Ranking: TF-IDF + SVD (LSA) vectors via scikit-learn, not sentence-transformers. No 1 GB PyTorch install or model download; good enough topic similarity on ~2,000 short summaries. `ranker.py` only needs the vectors file, so a neural embedding can replace `build_index.py`'s vectoriser later without touching the server.
8. Recall quiz: rule-based, no LLM. First sentence of a card with the subject blanked out; 3 wrong answers are similar articles from the same topic. Free, offline, no prompt-injection surface. ~85% of cards get a quiz.
9. Hosting: local only for V1 (works on a phone over home Wi-Fi). Deployment is a V2 decision.
10. Identity: no logins. Anonymous random ID in an HttpOnly cookie.

## Open — your call

11. Feed narrowing: after two long reads on Tech cards, 47 of the next 50 cards were Tech, even with AI & ML also selected. That's the ranker working as specified (80% similarity, 20% random), but it narrows fast. Options: (A) keep it; (B) guarantee each chosen topic a minimum share per page, e.g. 1 in 5; (C) raise exploration from 20% to 30–40%.
12. Seed trimming (decision 6, still to do): 196 of 400 Politics cards are "Politics of X", "Foreign relations of X" or "Government of X". Sports also has filler like "Sport in X", "Montana" and "Culture of Uganda".

## Changes made during the build (for the record)

- Seed sampling: the API returns links alphabetically, so a 400 cap kept only A-names. It now uses a seeded random sample, and "1990s in …" year pages are skipped.
- Politics outline: "Outline of politics" is a redirect, so collect now follows redirects.
- Fetching: switched from one request per article to batches of 20 via the Action API (~100 requests instead of ~2,000). Wikimedia was rate-limiting one-at-a-time requests to the point of taking hours.
- Thumbnails: Wikipedia now serves them from thumb.wikimedia.org as well as upload.wikimedia.org. Both are allowed (fetcher, API, frontend and CSP); everything else is dropped.
- Quiz: a quiz you scroll past is not shown again, and you can only answer a quiz you were shown, so the answer key can't be scraped.
- Security fixes from an independent review:
  - every request body is size-capped;
  - 422 errors no longer echo input;
  - each user's event history is capped and indexed;
  - huge integer ids give 422 instead of 500;
  - cross-site requests to /api are rejected;
  - the reading time for the card on screen is saved when a phone backgrounds the tab.
