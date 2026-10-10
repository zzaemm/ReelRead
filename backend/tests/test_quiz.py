"""Tests for quiz.py (pure functions, no DB)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quiz import first_sentence, make_question, make_quiz, strip_parentheticals, subject_of

DISTRACTORS = ["Java (programming language)", "Rust (programming language)", "Haskell"]


def card(title, extract):
    return {"title": title, "extract": extract}


# ----------------------------------------------------------------- masking variants
def test_disambiguator_removed_from_subject():
    assert subject_of("Python (programming language)") == "Python"
    assert subject_of("Haskell") == "Haskell"
    q = make_question("Python is a high-level, general-purpose programming language. More.",
                      "Python (programming language)")
    assert q == "___ is a high-level, general-purpose programming language."


def test_leading_the_is_swallowed():
    q = make_question("The Federal Reserve is the central bank of the United States. It ...",
                      "Federal Reserve")
    assert q == "___ is the central bank of the United States."


def test_title_starting_with_the():
    q = make_question("The Beatles were an English rock band formed in Liverpool in 1960.",
                      "The Beatles")
    assert q == "___ were an English rock band formed in Liverpool in 1960."


def test_case_insensitive_and_every_occurrence():
    q = make_question("MACHINE LEARNING is a field of study; machine learning powers apps.",
                      "Machine learning")
    assert q == "___ is a field of study; ___ powers apps."


def test_word_boundaries_and_plural():
    # "Go" must not be masked inside "Google" / "algorithm".
    q = make_question("Go is a programming language designed at Google for algorithms.",
                      "Go (programming language)")
    assert q == "___ is a programming language designed at Google for algorithms."
    q = make_question("Neural networks are machine learning models inspired by brains.",
                      "Neural network")
    assert q.startswith("___ are machine learning models")


def test_flexible_whitespace_and_curly_apostrophe():
    q = make_question("Moore’s law is the observation that transistor counts double.",
                      "Moore's law")
    assert q == "___ is the observation that transistor counts double."


# ----------------------------------------------------------------- parentheticals
def test_parentheticals_removed_including_nested():
    assert strip_parentheticals("Barack Obama (born August 4, 1961 (age 65)) is a politician.") \
        == "Barack Obama is a politician."
    assert strip_parentheticals("A (b (c (d) e) f) g [1].") == "A g."
    assert strip_parentheticals("Text (unclosed aside") is None


def test_pronunciation_with_dots_does_not_break_split():
    q = make_question("Python (/ˈpaɪθɒn/; c. 1991 – present. ok) is a "
                      "programming language used widely. Second.", "Python (programming language)")
    assert q == "___ is a programming language used widely."


# ----------------------------------------------------------------- sentence splitting
@pytest.mark.parametrize("text, expected", [
    ("J. R. R. Tolkien was an English writer and philologist. He wrote books.",
     "J. R. R. Tolkien was an English writer and philologist."),
    ("The U.S. Senate is the upper chamber of the U.S. Congress. It has 100 members.",
     "The U.S. Senate is the upper chamber of the U.S. Congress."),
    ("Apple Inc. is an American company. It makes phones.", "Apple Inc. is an American company."),
    ("Google is a subsidiary of Alphabet Inc. The company is big.",
     "Google is a subsidiary of Alphabet Inc."),
    ("St. Louis is a city where Dr. Smith lives, e.g. On weekends. Next.",
     "St. Louis is a city where Dr. Smith lives, e.g. On weekends."),
    ("It costs 3.5 dollars. Next.", "It costs 3.5 dollars."),
    ('He said "Hi." Then left.', 'He said "Hi."'),
    ("Is it? Yes.", "Is it?"),
    ("No terminal punctuation", "No terminal punctuation"),
])
def test_first_sentence(text, expected):
    assert first_sentence(text) == expected


def test_abbreviation_in_real_quiz():
    q = make_question("The Supreme Court of the U.S. is the highest court in the federal "
                      "judiciary. It has nine justices.", "Supreme Court of the United States")
    assert q is None   # subject spelled differently -> not found -> None
    q = make_question("Federal Reserve Bank of St. Louis is one of 12 regional banks of the "
                      "U.S. Federal Reserve System. More.", "Federal Reserve Bank of St. Louis")
    assert q == "___ is one of 12 regional banks of the U.S. Federal Reserve System."


# ----------------------------------------------------------------- None cases
def test_subject_not_found():
    assert make_question("Association football is a team sport played between two teams.",
                         "Lionel Messi") is None


def test_full_name_with_middle_names():
    assert make_question("Lionel Andr\u00e9s Messi is an Argentine professional footballer. He ...",
                         "Lionel Messi") == "___ is an Argentine professional footballer."
    assert make_question('Earvin "Magic" Johnson Jr. is an American former basketball player.',
                         "Magic Johnson") is None    # nickname-first names aren't guessed
    assert make_question('Stephen "Steph" Curry is an American professional basketball player.',
                         "Stephen Curry") == "___ is an American professional basketball player."
    # only at the start of the sentence, never mid-text
    assert make_question("The prize was won by Alan Mathison Turing, an English mathematician.",
                         "Alan Turing") is None


def test_too_short():
    assert make_question("Chess is a board game. It has rules.", "Chess") is None


def test_too_long():
    assert make_question("Chess is " + "very " * 100 + "old.", "Chess") is None


def test_answer_still_in_question(monkeypatch):
    import quiz
    assert quiz._leaks("___ is a moon of mercury.", "Mercury (planet)")
    assert quiz._leaks("___ orbits Mercury (planet).", "Mercury (planet)")
    assert not quiz._leaks("___ is a mercurial planet.", "Mercury (planet)")
    # Simulate a masking step that misses an occurrence: the guard must return None.
    monkeypatch.setattr(quiz, "mask_subject", lambda s, t: s.replace("Mercury", "___", 1))
    assert quiz.make_question("Mercury is the smallest planet, and Mercury is closest to the Sun.",
                              "Mercury (planet)") is None


def test_unbalanced_parens_and_empty_inputs():
    assert make_question("Chess (a board game is played by two players on a board.", "Chess") \
        is None
    assert make_question("", "Chess") is None
    assert make_quiz(card("", "Some extract that is long enough to be a question."),
                     DISTRACTORS, random.Random(0)) is None


def test_not_enough_distractors():
    c = card("Python (programming language)",
             "Python is a high-level, general-purpose programming language.")
    assert make_quiz(c, ["Java", "java", "Python (programming language)"], random.Random(0)) \
        is None
    assert make_quiz(c, [], random.Random(0)) is None


# ----------------------------------------------------------------- options
def test_options_distinct_with_correct_answer_index():
    c = card("Python (programming language)",
             "Python is a high-level, general-purpose programming language.")
    seen_positions = set()
    for seed in range(30):
        q = make_quiz(c, DISTRACTORS + ["Extra one"], random.Random(seed))
        assert q is not None
        assert set(q) == {"question", "options", "answer_index"}
        assert len(q["options"]) == 4 and len(set(q["options"])) == 4
        assert q["options"][q["answer_index"]] == "Python (programming language)"
        assert set(q["options"]) - {c["title"]} == set(DISTRACTORS)   # first 3 used
        seen_positions.add(q["answer_index"])
    assert len(seen_positions) > 1            # shuffled, not always in slot 0


def test_duplicate_distractors_are_skipped():
    c = card("Haskell", "Haskell is a purely functional programming language with lazy evaluation.")
    q = make_quiz(c, ["Rust", "rust", "HASKELL", "Java", "Go"], random.Random(1))
    assert sorted(q["options"]) == ["Go", "Haskell", "Java", "Rust"]


def test_same_rng_same_quiz():
    c = card("Haskell", "Haskell is a purely functional programming language with lazy evaluation.")
    assert make_quiz(c, ["A", "B", "C"], random.Random(5)) == \
        make_quiz(c, ["A", "B", "C"], random.Random(5))
