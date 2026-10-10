"""
Rule-based recall questions ("cloze" questions) made from a card's first sentence.

    "Python is a high-level programming language."  ->  "___ is a high-level programming language."

No ML here: just careful text clean-up. The rules (SPEC.md):

1. Take the first sentence of the extract (without breaking on "U.S.", "Dr.", "J. R. R." ...).
2. Remove parenthetical asides such as pronunciations and dates, including nested ones.
3. Replace the subject with ``___``, case-insensitively, at word boundaries. The subject is
   the title minus a trailing "(disambiguator)". A leading "The " is swallowed too, so
   "The Federal Reserve is ..." becomes "___ is ...".
4. Return None if the subject isn't found, the question is shorter than 40 chars, or the
   answer still appears in the question.
5. Options = answer + 3 distractor titles, shuffled with ``rng``.
"""
from __future__ import annotations

import random
import re
from typing import Mapping, Optional

BLANK = "___"
MIN_QUESTION_CHARS = 40
# Not in SPEC: a 600-char first sentence is unreadable on a phone and is usually a sign
# of a run-on or a failed sentence split. Treat it as "low quality".
MAX_QUESTION_CHARS = 400

# Abbreviations whose dot never ends a sentence, compared lower-case without the final
# dot. They are usually followed by a capital or digit ("Dr. Who", "St. Louis", "No. 1"),
# which is exactly what would otherwise look like a sentence boundary.
# Single-letter initials ("J. R. R.") and dotted acronyms ("U.S.", "e.g.", "a.k.a.") are
# handled by rules in first_sentence(), so they aren't listed here.
# Deliberately NOT listed: "Inc.", "Ltd.", "Co.", "etc." -- these can genuinely end a
# sentence ("...a subsidiary of Alphabet Inc. The company..."). Mid-sentence they are
# followed by a lower-case word ("Apple Inc. is ..."), which is never a boundary anyway.
_NEVER_END = {
    "mr", "mrs", "ms", "dr", "prof", "st", "mt", "ft", "jr", "sr", "sen", "rep", "gov",
    "gen", "col", "lt", "capt", "sgt", "rev", "hon", "pres", "no", "nos", "vs",
    "approx", "ca", "fl", "cf", "al", "vol", "pp", "op", "dept", "univ", "bros",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "ph.d",
}
_DOTTED = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")          # U.S / e.g / a.k.a (final dot stripped)
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")

_PARENS = re.compile(r"\s*\([^()]*\)")       # innermost (...) group with its leading space
_BRACKETS = re.compile(r"\s*\[[^\[\]]*\]")   # [1]-style citation markers, just in case
_DISAMBIG = re.compile(r"\s*\([^()]*\)\s*$")  # "Python (programming language)" -> "Python"


def first_sentence(text: str) -> str:
    """Return the first sentence of ``text``.

    A boundary is ``. ! ?`` (plus closing quotes/brackets), whitespace, then a capital
    letter or digit. A ``.`` boundary is ignored when the word before it is a single
    letter (an initial, as in "J. R. R. Tolkien"), a dotted acronym ("U.S."), or a known
    abbreviation ("Dr.", "St.", "e.g.").
    """
    text = text.strip()
    for m in _BOUNDARY.finditer(text):
        if text[m.start()] == ".":
            before = text[: m.start()].rsplit(None, 1)
            word = before[-1].lstrip("\"'(“[") if before else ""
            low = word.lower()
            if len(word) == 1 and word.isalpha():
                continue                                    # initial: "J. R. R. Tolkien"
            if _DOTTED.match(word) or low in _NEVER_END:
                continue                                    # "U.S. Senate", "Dr. Who"
        return text[: m.end()].strip()       # keeps a closing quote/bracket: '... "Hi."'
    return text


def strip_parentheticals(text: str) -> Optional[str]:
    """Remove ``(...)`` asides (nested ones too) and ``[...]`` markers.

    Nested parens are handled by repeatedly deleting the *innermost* group until none
    is left: "a (b (c) d) e" -> "a (b d) e" -> "a e". Returns None if unbalanced
    brackets remain (a truncated extract), because then we can't trust the text.
    """
    prev = None
    while prev != text:
        prev = text
        text = _PARENS.sub("", text)
        text = _BRACKETS.sub("", text)
    if any(ch in text for ch in "()[]"):
        return None
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)      # "Obama , born" -> "Obama, born"
    text = re.sub(r",\s*,", ",", text)
    return text.strip()


def subject_of(title: str) -> str:
    """"Python (programming language)" -> "Python"."""
    return _DISAMBIG.sub("", title).strip() or title.strip()


def _norm_quotes(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _subject_pattern(subject: str) -> str:
    """Regex for the subject as whole words: any whitespace between words, optional plural."""
    words = [re.escape(w) for w in subject.split()]
    return r"(?<!\w)" + r"\s+".join(words) + r"(?:e?s)?(?!\w)"


def mask_subject(sentence: str, title: str) -> Optional[str]:
    """Replace the subject in ``sentence`` with ``___``; None if it isn't there."""
    subject = _norm_quotes(subject_of(title))
    sentence = _norm_quotes(sentence)
    variants = [subject]
    if subject.lower().startswith("the ") and len(subject) > 4:
        variants.append(subject[4:])                 # title "The Guardian" -> also "Guardian"
    found = False
    for v in variants:                               # longest variant first
        pat = _subject_pattern(v)
        # "The X is ..." at the very start -> "___ is ..." (not "The ___ is ...").
        sentence, n1 = re.subn(r"^the\s+" + pat, BLANK, sentence, count=1, flags=re.IGNORECASE)
        sentence, n2 = re.subn(pat, BLANK, sentence, flags=re.IGNORECASE)
        found = found or bool(n1 or n2)
    if not found:
        sentence, found = _mask_full_name(sentence, subject)
    return sentence if found else None


# A capitalised middle name or a quoted nickname: "Mathison", "Andrés", '"Leo"', "J."
_MIDDLE_TOKEN = r"(?:[A-ZÀ-Þ][\w.'À-ɏ-]*|\"[^\"]{1,30}\")"


def _mask_full_name(sentence: str, subject: str) -> tuple[str, bool]:
    """Fallback for biographies: title "Alan Turing", lead "Alan Mathison Turing was ...".

    Only at the very start of the sentence, and only 1-3 capitalised middle tokens
    between the first word and the rest of the title, so we never swallow ordinary text.
    """
    words = subject.split()
    if len(words) < 2:
        return sentence, False
    pat = (r"^(?i:" + re.escape(words[0]) + r")(?:\s+" + _MIDDLE_TOKEN + r"){1,3}\s+(?i:"
           + r"\s+".join(re.escape(w) for w in words[1:]) + r")(?!\w)")
    new, n = re.subn(pat, BLANK, sentence, count=1)
    if n:   # mask any later mentions of the plain title too
        new = re.sub(_subject_pattern(subject), BLANK, new, flags=re.IGNORECASE)
    return new, bool(n)


def _leaks(question: str, title: str) -> bool:
    """True if the answer (full title or its subject) still appears as whole words."""
    q = _norm_quotes(question)
    for s in {_norm_quotes(title), _norm_quotes(subject_of(title))}:
        if s and re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", q, re.IGNORECASE):
            return True
    return False


def make_question(extract: str, title: str) -> Optional[str]:
    """Steps 1-4: extract -> cloze question string, or None if it isn't good enough."""
    if not extract or not title or not title.strip():
        return None
    # Parentheticals are removed *before* the sentence split: asides like
    # "(c. 1500 - d. 1560)" contain dots that would otherwise confuse the splitter.
    # The result is the same as split-then-strip, just more robust.
    cleaned = strip_parentheticals(extract)
    if not cleaned:
        return None
    sentence = first_sentence(cleaned)
    question = mask_subject(sentence, title)
    if question is None:
        return None
    question = re.sub(r"\s+", " ", question).strip()
    if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS):
        return None
    if question.replace(BLANK, "").strip(" .,;:") == "":
        return None
    if _leaks(question, title):
        return None
    return question


def make_quiz(card: Mapping, distractor_titles: list[str],
              rng: Optional[random.Random] = None) -> Optional[dict]:
    """Build ``{"question", "options": [4 titles], "answer_index"}`` or return None.

    ``card`` needs ``"title"`` and ``"extract"``. ``distractor_titles`` should hold at
    least 3 usable titles; ones equal to the answer (ignoring case) or repeated are
    skipped. Fewer than 3 usable distractors -> None.
    """
    rng = rng or random.Random()
    title = str(card["title"]).strip()
    question = make_question(str(card["extract"] or ""), title)
    if question is None:
        return None

    seen = {title.casefold()}
    distractors = []
    for d in distractor_titles:
        d = str(d).strip()
        if d and d.casefold() not in seen:
            seen.add(d.casefold())
            distractors.append(d)
        if len(distractors) == 3:
            break
    if len(distractors) < 3:
        return None

    options = [title] + distractors
    rng.shuffle(options)
    return {"question": question, "options": options, "answer_index": options.index(title)}
