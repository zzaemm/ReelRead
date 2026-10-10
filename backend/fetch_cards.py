"""
Build the card database from Wikipedia in two phases.

  python fetch_cards.py collect   # Outline pages -> seeds/<topic>.txt (one title per line)
  # ...open the seed files and delete any off-topic lines (decision 6)...
  python fetch_cards.py fetch     # seeds/*.txt -> summaries -> cards table in SQLite

Fetch is resumable and cheap to rerun: titles it has already resolved are not
downloaded again, so Ctrl+C is safe. Deleting lines from seeds/*.txt and
rerunning fetch removes those cards (no network needed).
"""
import random
import re
import sys
import time
from pathlib import Path

import requests

from db import connect, init_db

# --- Config -----------------------------------------------------------------

# Wikimedia blocks generic user agents. Put your own contact here before running.
USER_AGENT = "ReelRead/0.1 (student project; contact: https://github.com/zzaemm)"

# Topic -> Wikipedia "Outline of" hub pages to harvest links from.
TOPICS = {
    "tech":     ["Outline of technology", "Outline of computer science"],
    "sports":   ["Outline of sports"],
    "ai_ml":    ["Outline of artificial intelligence", "Outline of machine learning"],
    "finance":  ["Outline of finance"],
    "politics": ["Outline of politics"],
}

MAX_PER_TOPIC = 400         # cap so one huge outline doesn't dominate the feed
MIN_EXTRACT_CHARS = 200     # skip stubs too short to be worth a card
REQUEST_DELAY = 0.2         # seconds between requests; be polite to Wikimedia

# Link titles that are navigation pages, not articles.
SKIP_PREFIXES = ("List of", "Lists of", "Outline of", "Index of", "Glossary of", "Timeline of")
YEAR_PAGE = re.compile(r"^\d{3,4}s? in ")   # "1990s in science and technology"
SAMPLE_SEED = 42                              # fixed so the sample is reproducible

ACTION_API = "https://en.wikipedia.org/w/api.php"
SEEDS_DIR = Path(__file__).parent / "seeds"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT

MAX_RETRIES = 5


def polite_get(url: str, **kwargs) -> requests.Response:
    """GET that backs off when Wikipedia says 'too many requests' (429) or is briefly down (503)."""
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, timeout=15, **kwargs)
        if resp.status_code not in (429, 503):
            return resp
        # Retry-After is seconds to wait; fall back to exponential backoff: 2, 4, 8, 16...
        retry_after = resp.headers.get("Retry-After", "")
        wait = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
        wait = min(wait, 60)                  # never trust a server to make us sleep forever
        print(f"  ... rate-limited, waiting {wait}s")
        time.sleep(wait)
    return resp                               # caller's raise_for_status reports the final 429


# --- Phase 1: collect titles -------------------------------------------------

def outline_links(page_title: str) -> list[str]:
    """Every article (namespace 0) linked from one Wikipedia page, following pagination."""
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "titles": page_title, "redirects": "1", "prop": "links", "plnamespace": "0", "pllimit": "max",
    }
    titles = []
    while True:
        resp = polite_get(ACTION_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        for page in data["query"]["pages"]:
            if page.get("missing"):
                print(f"  ! page not found: {page_title}")
            titles += [link["title"] for link in page.get("links", [])]
        if "continue" not in data:
            return titles
        params.update(data["continue"])   # carries plcontinue to the next request
        time.sleep(REQUEST_DELAY)


def collect() -> None:
    SEEDS_DIR.mkdir(exist_ok=True)
    for topic, outlines in TOPICS.items():
        titles = []
        for outline in outlines:
            print(f"[{topic}] reading {outline}")
            titles += outline_links(outline)
        # dedupe while keeping order, drop navigation pages, apply cap
        seen, kept = set(), []
        for t in titles:
            if t not in seen and not t.startswith(SKIP_PREFIXES) and not YEAR_PAGE.match(t):
                seen.add(t)
                kept.append(t)
        # The API returns links alphabetically, so slicing would keep only A-names.
        # Sample randomly (fixed seed) instead, then sort so the file is easy to trim.
        if len(kept) > MAX_PER_TOPIC:
            kept = sorted(random.Random(SAMPLE_SEED).sample(kept, MAX_PER_TOPIC))
        (SEEDS_DIR / f"{topic}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"[{topic}] {len(kept)} titles -> seeds/{topic}.txt")
    print("\nNow open seeds/*.txt, delete off-topic lines, then run: python fetch_cards.py fetch")


# --- Phase 2: fetch summaries -------------------------------------------------
#
# One Action API request returns the intro text, thumbnail, URL and
# disambiguation flag for up to BATCH_SIZE titles, so ~2,000 titles take
# ~100 requests instead of 2,000. Far kinder to Wikimedia's rate limits.

BATCH_SIZE = 20             # TextExtracts returns at most 20 intros per request
MAX_EXTRACT_CHARS = 1100    # a card is a short read; longer intros are cut at a sentence end
# Wikimedia's image CDNs. Anything else is dropped so a poisoned response can't
# make users' browsers load arbitrary URLs.
THUMBNAIL_HOSTS = ("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")
ARTICLE_URL_PREFIX = "https://en.wikipedia.org/wiki/"


def strip_math(text: str) -> str:
    """Remove LaTeX blocks like '{\\displaystyle f(x)}' or '{\\textstyle ...}'.

    Wikipedia's plain-text extracts include the math twice: a readable version
    and then the raw LaTeX in braces. We keep the readable one. Braces nest, so
    this walks the string counting depth instead of using a regex.
    """
    out, i = [], 0
    markers = ("{\\displaystyle", "{\\textstyle")
    while True:
        hits = [j for j in (text.find(m, i) for m in markers) if j != -1]
        j = min(hits) if hits else -1
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1               # unbalanced braces: k ran to the end, rest is dropped
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).replace(" ,", ",").replace(" .", ".")


def clean_extract(raw: str) -> str:
    """Plain-text intro -> card text: no math markup, 1-3 paragraphs, length-capped."""
    paragraphs = [p.strip() for p in strip_math(raw).split("\n") if p.strip()]
    text = ""
    for p in paragraphs[:3]:
        text = f"{text}\n\n{p}" if text else p
        if len(text) >= 350:            # enough for one swipe; stop adding paragraphs
            break
    if len(text) > MAX_EXTRACT_CHARS:
        cut = text.rfind(". ", 0, MAX_EXTRACT_CHARS)
        text = text[: cut + 1] if cut > MIN_EXTRACT_CHARS else text[:MAX_EXTRACT_CHARS].rstrip() + "…"
    return text


def page_to_card(page: dict) -> dict | None:
    """One page object from the API -> card dict, or None if it shouldn't be a card."""
    if page.get("missing") or page.get("invalid") or page.get("ns", 0) != 0:
        return None
    if "disambiguation" in page.get("pageprops", {}):
        return None
    extract = clean_extract(page.get("extract") or "")   # plain text; HTML is never requested
    if len(extract) < MIN_EXTRACT_CHARS:
        return None
    url = page.get("fullurl", "")
    if not url.startswith(ARTICLE_URL_PREFIX):
        return None
    thumb = (page.get("thumbnail") or {}).get("source")
    if thumb and not thumb.startswith(THUMBNAIL_HOSTS):
        thumb = None
    return {"id": page["pageid"], "title": page["title"], "extract": extract,
            "url": url, "thumbnail": thumb}


def fetch_batch(titles: list[str]) -> dict[str, dict | None]:
    """Seed title -> card (or None if rejected) for up to BATCH_SIZE titles."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "titles": "|".join(titles),
        "prop": "extracts|pageimages|pageprops|info",
        "exintro": "1", "explaintext": "1", "exlimit": "max",
        "piprop": "thumbnail", "pithumbsize": "640", "pilimit": "max",
        "ppprop": "disambiguation", "inprop": "url",
    }
    pages: dict[str, dict] = {}
    renames: dict[str, str] = {}        # seed spelling -> normalised -> redirect target
    while True:                         # the API may split one answer over several replies
        resp = polite_get(ACTION_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        q = data.get("query", {})
        for step in q.get("normalized", []) + q.get("redirects", []):
            renames[step["from"]] = step["to"]
        for page in q.get("pages", []):
            merged = pages.setdefault(page["title"], {})
            merged.update({k: v for k, v in page.items() if k not in merged or not merged[k]})
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(REQUEST_DELAY)

    result = {}
    for title in titles:
        final, hops = title, 0
        while final in renames and hops < 5:      # follow normalise -> redirect chain
            final, hops = renames[final], hops + 1
        page = pages.get(final)
        result[title] = page_to_card(page) if page else None
    return result


def fetch() -> None:
    """
    Make the database match the seed files.

    The seed_titles table remembers how each seed line resolved (to a card id, or
    to NULL = rejected), so reruns only download titles they've never seen.
    At the end, any card/topic pair that's no longer in a seed file is removed:
    trimming a seed file and rerunning fetch is how you delete cards.
    """
    init_db()
    seed_files = sorted(SEEDS_DIR.glob("*.txt"))
    if not seed_files:
        sys.exit("No seed files found. Run: python fetch_cards.py collect")

    failed = 0
    with connect() as conn:
        resolved = {r["seed_title"]: r["card_id"] for r in conn.execute("SELECT * FROM seed_titles")}
        topic_titles = {
            seed.stem: list(dict.fromkeys(
                t.strip() for t in seed.read_text(encoding="utf-8").splitlines() if t.strip()))
            for seed in seed_files
        }
        new = list(dict.fromkeys(
            t for titles in topic_titles.values() for t in titles if t not in resolved))
        print(f"{sum(map(len, topic_titles.values()))} seed lines, {len(new)} titles to download "
              f"({-(-len(new) // BATCH_SIZE)} requests)")

        for start in range(0, len(new), BATCH_SIZE):
            batch = new[start:start + BATCH_SIZE]
            try:
                cards = fetch_batch(batch)
            except (requests.RequestException, ValueError, KeyError) as e:
                # network error or malformed JSON: leave unresolved so a rerun retries them
                print(f"  ! batch starting {batch[0]!r}: {e}")
                failed += len(batch)
                continue
            for title, card in cards.items():
                if card is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO cards (id, title, extract, url, thumbnail) "
                        "VALUES (:id, :title, :extract, :url, :thumbnail)", card)
                card_id = card["id"] if card else None
                conn.execute("INSERT OR REPLACE INTO seed_titles VALUES (?, ?)", (title, card_id))
                resolved[title] = card_id
            conn.commit()                                 # checkpoint every batch
            done = min(start + BATCH_SIZE, len(new))
            if done % 200 == 0 or done == len(new):
                print(f"  {done}/{len(new)}")
            time.sleep(REQUEST_DELAY)

        # Sync topic tags to exactly what the seed files say, then drop untagged cards.
        wanted = sorted({(resolved[t], topic) for topic, titles in topic_titles.items()
                         for t in titles if resolved.get(t) is not None})
        conn.execute("DELETE FROM card_topics")
        conn.executemany("INSERT OR IGNORE INTO card_topics VALUES (?, ?)", wanted)
        removed = conn.execute(
            "DELETE FROM cards WHERE id NOT IN (SELECT card_id FROM card_topics)").rowcount
        # forget resolutions pointing at deleted cards, so re-adding the line re-downloads it
        conn.execute("DELETE FROM seed_titles WHERE card_id IS NOT NULL "
                     "AND card_id NOT IN (SELECT id FROM cards)")
        conn.commit()

        print("\nCards per topic:")
        for row in conn.execute("SELECT topic, COUNT(*) n FROM card_topics GROUP BY topic ORDER BY topic"):
            print(f"  {row['topic']:<10} {row['n']}")
        total = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        print(f"Done. {total} cards total, {removed} removed (trimmed from seeds).")
        if failed:
            print(f"{failed} titles failed to download; rerun fetch to retry them.")
        print("Next: python build_index.py")


if __name__ == "__main__":
    if "YOUR_EMAIL" in USER_AGENT:
        sys.exit("Set USER_AGENT at the top of fetch_cards.py first (Wikimedia requires a contact).")
    commands = {"collect": collect, "fetch": fetch}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Usage: python fetch_cards.py [collect|fetch]")
    commands[sys.argv[1]]()
