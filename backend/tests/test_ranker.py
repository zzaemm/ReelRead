"""Tests for ranker.py and build_index.py, using synthetic vectors and a temp DB only."""
import os
import random
import sys
import tempfile
import time
from pathlib import Path

# Point db.py at a throwaway file BEFORE anything imports it.
os.environ.setdefault("REELREAD_DB", str(Path(tempfile.mkdtemp()) / "test_engine.db"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from ranker import Ranker, Signal, _interleave, signal_weights


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def toy():
    """6 cards in 3 dims: 1-2 point along x, 3-4 along y, 5-6 along z."""
    ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    vecs = np.stack([unit(1, 0.1, 0), unit(1, 0, 0.1), unit(0.1, 1, 0),
                     unit(0, 1, 0.1), unit(0.1, 0, 1), unit(0, 0.1, 1)])
    return Ranker(ids, vecs)


# ----------------------------------------------------------------- signal weights
def test_view_weight_scales_with_dwell_and_caps_at_one():
    w = signal_weights([Signal(1, "view", 5000), Signal(2, "view", 60000)])
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(1.0)


def test_short_view_is_a_skip():
    assert signal_weights([Signal(1, "view", 1499)])[1] == pytest.approx(-0.5)
    assert signal_weights([Signal(1, "view", 1500)])[1] == pytest.approx(0.15)


def test_like_and_unlike():
    assert signal_weights([Signal(1, "like", None)])[1] == pytest.approx(1.0)
    # like + unlike cancel out
    assert signal_weights([Signal(1, "like", None), Signal(1, "unlike", None)]).get(1, 0) == 0
    # two likes, one unlike -> one like left
    sigs = [Signal(1, "like", None), Signal(1, "like", None), Signal(1, "unlike", None)]
    assert signal_weights(sigs)[1] == pytest.approx(1.0)
    # an unlike with no like to cancel never goes negative
    assert signal_weights([Signal(1, "unlike", None)]).get(1, 0) == 0


def test_weights_add_up_per_card():
    sigs = [Signal(1, "view", 10000), Signal(1, "like", None), Signal(1, "view", 500)]
    assert signal_weights(sigs)[1] == pytest.approx(1.0 + 1.0 - 0.5)


def test_view_without_dwell_and_unknown_kinds_ignored():
    assert signal_weights([Signal(1, "view", None), Signal(2, "quiz_correct", None)]) == {}


def test_skip_pushes_ranking_away(toy):
    # Skipping an x-card should rank x-cards last.
    out = toy.rank([Signal(1, "view", 200)], [2, 3, 4, 5, 6], k=5, explore_frac=0.0)
    assert out[-1] == 2


def test_like_pulls_ranking_towards(toy):
    out = toy.rank([Signal(3, "like", None)], [1, 2, 4, 5, 6], k=2, explore_frac=0.0)
    assert out[0] == 4


# ----------------------------------------------------------------- rank()
def test_cold_start_is_random_sample(toy):
    cands = [1, 2, 3, 4, 5, 6]
    a = toy.rank([], cands, k=6, rng=random.Random(1))
    b = toy.rank([], cands, k=6, rng=random.Random(2))
    assert sorted(a) == cands and sorted(b) == cands
    assert a != b    # different seeds -> different orders (fixed seeds, so deterministic)


def test_cancelled_signals_count_as_cold_start(toy):
    sigs = [Signal(1, "like", None), Signal(1, "unlike", None)]
    out = toy.rank(sigs, [1, 2, 3, 4, 5, 6], k=3, rng=random.Random(0))
    assert len(out) == 3 and set(out) <= {1, 2, 3, 4, 5, 6}


def test_explore_fraction_count():
    rng = np.random.default_rng(0)
    n = 200
    ids = np.arange(1, n + 1, dtype=np.int64)
    vecs = rng.normal(size=(n, 16)).astype(np.float32)
    r = Ranker(ids, vecs)
    sigs = [Signal(1, "like", None)]
    cands = list(range(2, n + 1))
    # exploit set = the top round(k * 0.8) = 8 cosine matches
    top8 = r.rank(sigs, cands, k=8, explore_frac=0.0)
    out = r.rank(sigs, cands, k=10, explore_frac=0.2, rng=random.Random(3))
    assert len(out) == 10
    assert set(top8) <= set(out)
    explore = [c for c in out if c not in top8]
    assert len(explore) == 2
    # interleaved, not appended: the explore picks are not both at the end
    assert sorted(out.index(c) for c in explore) != [8, 9]
    assert out[0] == top8[0]


def test_interleave_spreads_explore():
    assert _interleave([1, 2, 3, 4, 5, 6, 7, 8], ["a", "b"]) == [1, 2, 3, "a", 4, 5, "b", 6, 7, 8]
    assert _interleave([], ["a", "b"]) == ["a", "b"]
    assert _interleave([1, 2], []) == [1, 2]


def test_explore_frac_extremes(toy):
    sigs = [Signal(1, "like", None)]
    assert len(toy.rank(sigs, [2, 3, 4, 5, 6], k=4, explore_frac=1.0, rng=random.Random(0))) == 4
    assert toy.rank(sigs, [2, 3, 4, 5, 6], k=1, explore_frac=0.0) == [2]


def test_distinct_ids_subset_of_candidates(toy):
    cands = [6, 5, 5, 4, 3, 3, 99, 1]
    for seed in range(20):
        out = toy.rank([Signal(1, "view", 8000)], cands, k=10, rng=random.Random(seed))
        assert len(out) == len(set(out)) == len(set(cands))
        assert set(out) <= set(cands)


def test_k_bigger_than_candidates_and_empty(toy):
    assert sorted(toy.rank([Signal(1, "like", None)], [3, 4], k=10)) == [3, 4]
    assert toy.rank([Signal(1, "like", None)], [], k=10) == []
    assert toy.rank([], [1, 2], k=0) == []


def test_unknown_candidate_ids_score_zero(toy):
    # Like an x-card, skip a z-card. 1001 has no vector -> score 0, which loses to
    # the matching x-card (positive) and beats the other z-card (negative).
    sigs = [Signal(1, "like", None), Signal(5, "view", 100)]
    out = toy.rank(sigs, [6, 1001, 2], k=3, explore_frac=0.0)
    assert out == [2, 1001, 6]


def test_signals_on_unknown_cards_are_cold_start(toy):
    out = toy.rank([Signal(999, "like", None)], [1, 2, 3], k=3, rng=random.Random(0))
    assert sorted(out) == [1, 2, 3]


# ----------------------------------------------------------------- loading
def test_missing_index_file_gives_random_ranker(tmp_path, caplog):
    r = Ranker.load(tmp_path / "nope.npz")
    assert len(r) == 0
    assert "not found" in caplog.text
    out = r.rank([Signal(1, "like", None)], [1, 2, 3, 4], k=3, rng=random.Random(0))
    assert len(out) == 3 and set(out) <= {1, 2, 3, 4}
    assert r.neighbours(1, 5) == []


def test_corrupt_index_file_gives_random_ranker(tmp_path, caplog):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"this is not a zip file")
    assert len(Ranker.load(bad)) == 0
    wrong_shape = tmp_path / "wrong.npz"
    np.savez(wrong_shape, card_ids=np.arange(3), vectors=np.zeros((5, 4)))
    assert len(Ranker.load(wrong_shape)) == 0
    missing_key = tmp_path / "missing.npz"
    np.savez(missing_key, card_ids=np.arange(3))
    assert len(Ranker.load(missing_key)) == 0
    assert "random" in caplog.text


def test_load_round_trip(tmp_path, toy):
    p = tmp_path / "index.npz"
    np.savez(p, card_ids=toy.card_ids, vectors=toy.vectors)
    r = Ranker.load(p)
    assert len(r) == 6 and r.dims == 3
    assert r.neighbours(1, 1) == [2]


# ----------------------------------------------------------------- neighbours
def test_neighbours_excludes_self_and_is_sorted(toy):
    nb = toy.neighbours(1, 5)
    assert 1 not in nb and len(nb) == 5
    assert nb[0] == 2 and nb[1] == 3 and nb[-1] == 6     # x-twin, then nearest y-card, ...
    assert toy.neighbours(3, 1) == [4]


def test_neighbours_edge_cases(toy):
    assert toy.neighbours(12345, 3) == []
    assert toy.neighbours(1, 0) == []
    assert len(toy.neighbours(1, 100)) == 5


# ----------------------------------------------------------------- speed
def test_rank_5000_candidates_under_50ms():
    rng = np.random.default_rng(0)
    n, d = 6000, 128
    r = Ranker(np.arange(1, n + 1, dtype=np.int64), rng.normal(size=(n, d)).astype(np.float32))
    sigs = [Signal(int(i), "view", int(rng.integers(0, 20000))) for i in rng.integers(1, n, 300)]
    sigs += [Signal(int(i), "like", None) for i in rng.integers(1, n, 30)]
    cands = [int(x) for x in rng.permutation(np.arange(1, n + 1))[:5000]]
    r.rank(sigs, cands, k=20)                    # warm-up
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        out = r.rank(sigs, cands, k=20, rng=random.Random(0))
        best = min(best, time.perf_counter() - t0)
    assert len(out) == 20
    assert best < 0.050, f"rank took {best * 1000:.1f} ms"


# ----------------------------------------------------------------- build_index (temp DB)
def test_build_index_end_to_end(tmp_path, monkeypatch):
    import db
    import build_index

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "build.db")
    db.init_db()
    themes = {
        "sports": ["football", "goal", "league", "match", "team", "player", "stadium", "season"],
        "finance": ["bank", "money", "interest", "market", "stock", "bond", "loan", "credit"],
        "tech": ["software", "computer", "code", "program", "network", "data", "chip", "server"],
    }
    rnd = random.Random(0)
    cid = 1
    with db.connect() as conn:
        for topic, words in themes.items():
            for i in range(15):
                title = f"{topic.title()} Thing {chr(65 + i)}{cid}"
                body = " ".join(rnd.choices(words, k=12))
                extract = f"{title} is a well known example of {body} in the field. It is notable."
                conn.execute("INSERT INTO cards (id, title, extract, url) VALUES (?,?,?,?)",
                             (cid, title, extract, f"https://en.wikipedia.org/wiki/{cid}"))
                conn.execute("INSERT INTO card_topics VALUES (?, ?)", (cid, topic))
                cid += 1
        conn.execute("INSERT INTO quizzes VALUES (1, 'stale question here', '[]', 0)")

    index = tmp_path / "idx.npz"
    assert build_index.main(["--index", str(index)]) == 0

    r = Ranker.load(index)
    assert len(r) == 45 and 1 <= r.dims <= 44
    # neighbours of a sports card are mostly sports cards (ids 1..15)
    nb = r.neighbours(1, 5)
    assert sum(1 <= x <= 15 for x in nb) >= 4

    import json
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM quizzes").fetchall()
        titles = {c["id"]: c["title"] for c in conn.execute("SELECT id, title FROM cards")}
    assert len(rows) == 45            # stale row replaced, every card got a quiz
    for row in rows:
        opts = json.loads(row["options"])
        assert len(opts) == 4 and len(set(opts)) == 4
        assert opts[row["answer_index"]] == titles[row["card_id"]]
        assert "___" in row["question"]
        assert titles[row["card_id"]].lower() not in row["question"].lower()


def test_build_index_too_few_cards(tmp_path, monkeypatch, capsys):
    import db
    import build_index

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tiny.db")
    db.init_db()
    with db.connect() as conn:
        conn.execute("INSERT INTO cards (id, title, extract, url) VALUES (1,'A','a b c','u')")
    assert build_index.main(["--index", str(tmp_path / "x.npz")]) == 1
    assert "need at least" in capsys.readouterr().err
    assert not (tmp_path / "x.npz").exists()
