"""
Offline step, run after fetch_cards.py:

    python build_index.py [--index PATH]

1. Turns every card into a dense vector and saves index.npz (used by ranker.py).
2. Rebuilds the quizzes table (one recall question per card, where possible).

The vector pipeline (a classic "LSA" / Latent Semantic Analysis setup):

* **TF-IDF**: each card becomes a sparse vector with one entry per vocabulary word.
  ``tf`` = how often the word appears in this card (``sublinear_tf`` uses 1 + log(tf)
  so a word repeated 20 times doesn't count 20x); ``idf = log(N / df)`` down-weights
  words that appear in many cards. Words in < 2 cards (``min_df=2``: typos, one-offs)
  or > 50% of cards (``max_df=0.5``: too generic) are dropped, as are English
  stop words.
* **TruncatedSVD**: factorises the (cards x words) matrix and keeps the top D
  directions of variation ("latent topics"). Words that co-occur get merged, so
  "neural network" and "deep learning" cards end up close even if they share few
  exact words. D = 128, or fewer for a small corpus.
* **L2-normalise**: scale every vector to length 1, so a plain dot product is the
  cosine similarity.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

import db
from quiz import make_quiz, subject_of
from ranker import Ranker

DEFAULT_INDEX = Path(__file__).parent / "index.npz"
MAX_DIMS = 128
RANDOM_STATE = 42          # fixed so the index (and the quizzes) are reproducible
MIN_CARDS = 3              # SVD needs n_components <= n_cards - 1, and >= 1 dimension
DISTRACTOR_BAND = (0, 40)  # only [1] is used: how many nearest neighbours to consider


# --------------------------------------------------------------------------- vectors
def fit_vectors(texts: list[str]) -> np.ndarray:
    """TF-IDF -> TruncatedSVD -> L2-normalise. Returns float32 (N x D)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.5)
    try:
        tfidf = TfidfVectorizer(**params).fit_transform(texts)
    except ValueError:
        # Tiny corpora can prune every term (e.g. 3 cards with no word in 2 of them).
        print("note: corpus too small for min_df=2/max_df=0.5; relaxing to min_df=1/max_df=1.0")
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)

    n_cards, n_features = tfidf.shape
    dims = min(MAX_DIMS, n_features - 1, n_cards - 1)
    if dims < 1:
        raise SystemExit(f"error: only {n_features} usable terms across {n_cards} cards; "
                         "fetch more cards before building the index")
    svd = TruncatedSVD(n_components=dims, algorithm="randomized", random_state=RANDOM_STATE)
    vecs = svd.fit_transform(tfidf).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms > 0)
    print(f"tf-idf: {n_features} terms; svd: {dims} dims, "
          f"{svd.explained_variance_ratio_.sum():.0%} of variance kept")
    return vecs


def save_index(path: Path, card_ids: np.ndarray, vectors: np.ndarray) -> None:
    """Write to a temp file then rename, so a running server never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:              # file handle: np.savez won't append ".npz"
        np.savez(f, card_ids=card_ids.astype(np.int64), vectors=vectors.astype(np.float32))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- quizzes
def _too_close(a: str, b: str) -> bool:
    """True if one title is (or contains) the other, ignoring case and disambiguators.

    Stops "Python" vs "Python (programming language)" or "Machine learning" vs
    "Quantum machine learning" appearing as answer + distractor.
    """
    a, b = subject_of(a).casefold(), subject_of(b).casefold()

    def contains(outer: str, inner: str) -> bool:      # whole words: "ai" is not in "chain"
        return re.search(r"(?<!\w)" + re.escape(inner) + r"(?!\w)", outer) is not None

    return a == b or contains(a, b) or contains(b, a)


def pick_distractors(card_id: int, ranker: Ranker, titles: dict[int, str],
                     topic_mates: list[int], rng: random.Random) -> list[str]:
    """3 plausible wrong answers for this card's quiz.

    Preference order:
      1. the nearest neighbours (top 40) that share a topic: pick 3 at random from
         the 6 closest that pass the filters, so options are related but varied;
      2. any of the top-40 neighbours;
      3. random cards sharing a topic.
    Filters: never equal to / containing / contained in the answer, all distinct.
    (Real-data check showed a random 5-40 band let in off-topic options like
    "Hawaii" for "Nuclear weapon", which made quizzes trivially easy.)
    """
    answer = titles[card_id]
    chosen: list[str] = []

    def ok(cid: int) -> bool:
        t = titles.get(cid)
        return bool(t) and cid != card_id and not _too_close(t, answer) \
            and not any(_too_close(t, c) for c in chosen)

    def try_add(pool: list[int]) -> None:
        for cid in pool:
            if len(chosen) == 3:
                return
            if ok(cid):
                chosen.append(titles[cid])

    near = ranker.neighbours(card_id, DISTRACTOR_BAND[1])
    mates = set(topic_mates)
    # closest same-topic neighbours that would be valid on their own
    shortlist = []
    for cid in near:
        if cid in mates and ok(cid) and not any(_too_close(titles[cid], titles[s]) for s in shortlist):
            shortlist.append(cid)
            if len(shortlist) == 6:
                break
    try_add(rng.sample(shortlist, len(shortlist)))
    if len(chosen) < 3:
        try_add(near)
    if len(chosen) < 3:
        try_add(rng.sample(topic_mates, min(len(topic_mates), 200)))
    return chosen


def build_quizzes(conn, cards, ranker: Ranker) -> int:
    """Delete and rebuild the quizzes table in one transaction. Returns rows inserted."""
    rng = random.Random(RANDOM_STATE)
    titles = {c["id"]: c["title"] for c in cards}
    topics_of: dict[int, list[str]] = {}
    by_topic: dict[str, list[int]] = {}
    for r in conn.execute("SELECT card_id, topic FROM card_topics"):
        topics_of.setdefault(r["card_id"], []).append(r["topic"])
        by_topic.setdefault(r["topic"], []).append(r["card_id"])

    rows = []
    for c in cards:
        mates = sorted({m for t in topics_of.get(c["id"], []) for m in by_topic[t]} - {c["id"]})
        distractors = pick_distractors(c["id"], ranker, titles, mates, rng)
        q = make_quiz(c, distractors, rng)
        if q is not None:
            rows.append((c["id"], q["question"], json.dumps(q["options"]), q["answer_index"]))

    with conn:   # one transaction: readers see either the old table or the new one
        conn.execute("DELETE FROM quizzes")
        conn.executemany(
            "INSERT INTO quizzes (card_id, question, options, answer_index) VALUES (?, ?, ?, ?)",
            rows)
    return len(rows)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build index.npz and the quizzes table.")
    ap.add_argument("--index", type=Path,
                    default=Path(os.environ.get("REELREAD_INDEX", DEFAULT_INDEX)),
                    help="output path (default: env REELREAD_INDEX or backend/index.npz)")
    args = ap.parse_args(argv)

    db.init_db()
    conn = db.connect()
    try:
        cards = conn.execute("SELECT id, title, extract FROM cards ORDER BY id").fetchall()
        if len(cards) < MIN_CARDS:
            print(f"error: {len(cards)} cards in {db.DB_PATH}; need at least {MIN_CARDS}. "
                  "Run fetch_cards.py first.", file=sys.stderr)
            return 1

        card_ids = np.array([c["id"] for c in cards], dtype=np.int64)
        vectors = fit_vectors([f'{c["title"]}. {c["extract"]}' for c in cards])
        save_index(args.index, card_ids, vectors)

        n_quizzes = build_quizzes(conn, cards, Ranker(card_ids, vectors))
    finally:
        conn.close()

    print(f"cards indexed: {len(cards)}  dims: {vectors.shape[1]}  -> {args.index}")
    print(f"quizzes built: {n_quizzes} ({n_quizzes / len(cards):.0%} of cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
