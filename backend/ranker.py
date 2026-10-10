"""
User-vector ranking for the ReelRead feed.

How it works (the short version)
--------------------------------
Offline, ``build_index.py`` turns every card into a dense vector of ~128
numbers (TF-IDF -> SVD/LSA -> L2-normalise). Cards about similar things end up
pointing in similar directions, so the *cosine similarity* between two card
vectors (the dot product of two unit vectors, ranging -1..1) tells us how
related they are.

At request time we build a *user vector* = a weighted sum of the vectors of
cards the user interacted with:

    u = sum_i  weight(signal_i) * v(card_i)

Lingering or liking pulls ``u`` towards a card's direction; a quick skip
pushes it away. To rank candidates we score each one with
``cosine(u, v_card) = (u / |u|) . v_card`` (card vectors are already unit
length) and take the best — plus a slice of random "explore" picks so the feed
doesn't collapse into one niche and so we keep learning about new interests.

Runtime dependency: numpy only.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

log = logging.getLogger("reelread.ranker")

# --- Signal weights (see SPEC.md "Engine interface") -------------------------
FULL_DWELL_MS = 10_000      # a view this long (or longer) counts as weight 1.0
SKIP_DWELL_MS = 1_500       # a view shorter than this is a skip ...
SKIP_WEIGHT = -0.5          # ... and pushes the user vector away from the card
LIKE_WEIGHT = 1.0           # each un-cancelled like


@dataclass
class Signal:
    """One thing the user did to one card (built by main.py from the events table)."""
    card_id: int
    kind: str          # "view" | "like" | "unlike"
    dwell_ms: Optional[int]


def signal_weights(signals: list[Signal]) -> dict[int, float]:
    """Collapse a list of signals into one total weight per card.

    * view:   ``min(dwell_ms / 10000, 1.0)``; if ``dwell_ms < 1500`` it is a skip: ``-0.5``.
              A view without a dwell time carries no information and is ignored.
    * like:   ``+1.0``
    * unlike: cancels one like (``-1.0``). Likes net of unlikes are clamped at 0
              per card, so an unlike can never push a card *below* neutral (e.g. if
              the matching like fell outside the 300-event window). Never-liked +
              unliked = 0, the same rule main.py uses for the ``liked`` flag.
    Unknown kinds (e.g. quiz events) are ignored.
    """
    view_w: dict[int, float] = {}
    net_likes: dict[int, int] = {}
    for s in signals:
        cid = int(s.card_id)
        if s.kind == "view":
            if s.dwell_ms is None:
                continue
            dwell = max(int(s.dwell_ms), 0)
            w = SKIP_WEIGHT if dwell < SKIP_DWELL_MS else min(dwell / FULL_DWELL_MS, 1.0)
            view_w[cid] = view_w.get(cid, 0.0) + w
        elif s.kind == "like":
            net_likes[cid] = net_likes.get(cid, 0) + 1
        elif s.kind == "unlike":
            net_likes[cid] = net_likes.get(cid, 0) - 1

    weights = dict(view_w)
    for cid, n in net_likes.items():
        if n > 0:
            weights[cid] = weights.get(cid, 0.0) + LIKE_WEIGHT * n
    return weights


def _interleave(exploit: list[int], explore: list[int]) -> list[int]:
    """Spread the explore picks evenly through the exploit list.

    With k slots and e explore items, explore item i goes at slot
    ``floor((i + 1) * k / (e + 1))``. Slots are >= 1 apart when e < k, so they
    never collide, and slot 0 is always the best exploit match.
    E.g. 8 exploit + 2 explore -> explore at positions 3 and 6.
    """
    if not exploit or not explore:
        return exploit + explore
    k, e = len(exploit) + len(explore), len(explore)
    slots = {(i + 1) * k // (e + 1): i for i in range(e)}
    out, it = [], iter(exploit)
    for pos in range(k):
        out.append(explore[slots[pos]] if pos in slots else next(it))
    return out


class Ranker:
    """Holds the card vectors and ranks candidates for a user.

    ``card_ids`` (int64, N) and ``vectors`` (float32, N x D) come from index.npz.
    An empty Ranker (no index) still works: every ranking is a random sample.
    """

    def __init__(self, card_ids=None, vectors=None):
        if card_ids is None or vectors is None or len(card_ids) == 0:
            self.card_ids = np.zeros(0, dtype=np.int64)
            self.vectors = np.zeros((0, 0), dtype=np.float32)
        else:
            ids = np.asarray(card_ids, dtype=np.int64).ravel()
            vecs = np.asarray(vectors, dtype=np.float32)
            if vecs.ndim != 2 or vecs.shape[0] != ids.shape[0]:
                raise ValueError(f"shape mismatch: {ids.shape} ids vs {vecs.shape} vectors")
            if not np.all(np.isfinite(vecs)):
                raise ValueError("vectors contain NaN/inf")
            # Re-normalise defensively so a dot product really is a cosine.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms > 0)
            self.card_ids, self.vectors = ids, vecs

        # card_id -> row lookup, built once. We keep the ids sorted so that a
        # whole batch of candidate ids can be looked up with one vectorised
        # np.searchsorted call instead of a Python loop.
        self._order = np.argsort(self.card_ids, kind="stable")
        self._sorted_ids = self.card_ids[self._order]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Ranker":
        """Load index.npz. A missing or corrupt file gives an empty (random) Ranker."""
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                ranker = cls(data["card_ids"], data["vectors"])
        except FileNotFoundError:
            log.warning("index file %s not found; ranking will be random", path)
            return cls()
        except Exception as exc:  # corrupt zip, missing arrays, bad shapes, ...
            log.warning("could not load index %s (%s); ranking will be random", path, exc)
            return cls()
        log.info("loaded index %s: %d cards x %d dims", path, *ranker.vectors.shape)
        return ranker

    def __len__(self) -> int:
        return int(self.card_ids.shape[0])

    @property
    def dims(self) -> int:
        return int(self.vectors.shape[1]) if len(self) else 0

    # ------------------------------------------------------------------ helpers
    def rows_for(self, ids) -> np.ndarray:
        """Vectorised card_id -> row index; -1 for ids not in the index."""
        ids = np.asarray(ids, dtype=np.int64).ravel()
        if len(self) == 0 or ids.size == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self._sorted_ids, ids)
        pos = np.minimum(pos, len(self) - 1)
        found = self._sorted_ids[pos] == ids
        return np.where(found, self._order[pos], -1)

    def user_vector(self, signals: list[Signal]) -> Optional[np.ndarray]:
        """``u = sum(weight * card_vector)``, normalised to unit length.

        Returns None when there is nothing to go on (no index, no signals on
        indexed cards, or the weights cancel out to a zero vector).
        """
        if len(self) == 0 or not signals:
            return None
        weights = signal_weights(signals)
        if not weights:
            return None
        rows = self.rows_for(list(weights.keys()))
        w = np.fromiter(weights.values(), dtype=np.float32, count=len(weights))
        keep = rows >= 0
        if not keep.any():
            return None
        u = w[keep] @ self.vectors[rows[keep]]          # (m,) @ (m, D) -> (D,)
        norm = float(np.linalg.norm(u))
        if norm < 1e-9:
            return None
        return u / norm

    # ------------------------------------------------------------------ ranking
    def rank(self, signals: list[Signal], candidate_ids: list[int], k: int,
             explore_frac: float = 0.2, rng: Optional[random.Random] = None) -> list[int]:
        """Pick up to ``k`` distinct ids from ``candidate_ids``.

        * Cold start (no usable user vector) or empty index -> random sample.
        * Otherwise ``round(k * (1 - explore_frac))`` "exploit" ids are the top
          cosine matches, the rest are random "explore" ids from the remaining
          candidates, and the two groups are interleaved.
        * Candidates without a vector score 0 (neutral: better than cards the
          user skipped, worse than cards like the ones they enjoyed).
        """
        rng = rng or random.Random()
        cands = list(dict.fromkeys(int(c) for c in candidate_ids))   # dedupe, keep order
        k = min(int(k), len(cands))
        if k <= 0:
            return []

        u = self.user_vector(signals)
        if u is None:
            return rng.sample(cands, k)

        explore_frac = min(max(float(explore_frac), 0.0), 1.0)
        n_exploit = min(round(k * (1 - explore_frac)), k)
        n_explore = k - n_exploit

        cand_arr = np.asarray(cands, dtype=np.int64)
        rows = self.rows_for(cand_arr)
        scores = np.zeros(len(cands), dtype=np.float32)
        has_vec = rows >= 0
        scores[has_vec] = self.vectors[rows[has_vec]] @ u      # cosine similarity

        # Shuffle first, then stable-sort by score: ties (e.g. all the 0-score
        # cards without vectors) come out in random order instead of id order.
        np_rng = np.random.default_rng(rng.getrandbits(64))
        perm = np_rng.permutation(len(cands))
        order = perm[np.argsort(-scores[perm], kind="stable")]

        exploit = [cands[i] for i in order[:n_exploit]]
        rest = [cands[i] for i in order[n_exploit:]]
        explore = rng.sample(rest, n_explore)
        return _interleave(exploit, explore)

    def neighbours(self, card_id: int, n: int) -> list[int]:
        """The ``n`` most similar cards by cosine, most similar first, excluding itself.

        Unknown ids and cards whose vector is all zeros return [].
        """
        if n <= 0 or len(self) == 0:
            return []
        row = int(self.rows_for([card_id])[0])
        if row < 0:
            return []
        v = self.vectors[row]
        if not v.any():
            return []
        sims = self.vectors @ v
        sims[row] = -np.inf                        # never return the card itself
        n = min(n, len(self) - 1)
        if n <= 0:
            return []
        top = np.argpartition(-sims, n - 1)[:n]   # O(N) top-n, unordered ...
        top = top[np.argsort(-sims[top], kind="stable")]   # ... then sort just those
        return [int(x) for x in self.card_ids[top]]
