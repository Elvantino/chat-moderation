import pytest

from src.preprocessing import (LexiconMatcher, deleet, normalize_text, osa_distance,
                               reduce_repeats)


@pytest.mark.parametrize("raw, expected", [
    ("noooo", "no"),
    ("fuuuuuck", "fuck"),
    ("cool", "cool"),                       # double letters preserved
])
def test_reduce_repeats(raw, expected):
    assert reduce_repeats(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("h4t3", "hate"),
    ("sh!t", "shit"),
    ("@$$", "@$$"),                         # no letters -> untouched
    ("2024", "2024"),
    ("idiot!", "idiot!"),                   # trailing "!" is punctuation, not leet
])
def test_deleet(raw, expected):
    assert deleet(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Y0u are a  STUP1D   iiiiidiooot!!!", "you are a stupid idiot"),
    ("check http://spam.com now", "check now"),
    ("from 192.168.0.1 hello", "from hello"),
    ("f.u.c.k off", "fuck off"),
    ("Don't", "dont"),
    ("Café naïve", "cafe naive"),
    ("", ""),
    (None, ""),
    ("I scored 50 points", "i scored 50 points"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_osa_distance():
    assert osa_distance("fcuk", "fuck") == 1      # transposition
    assert osa_distance("kitten", "sitting") == 3
    assert osa_distance("abc", "abc") == 0
    assert osa_distance("a", "abcdef", max_dist=1) == 2


@pytest.fixture
def lex():
    return LexiconMatcher(["fuck*", "idiot*", "shit*", "loser*", "kill yourself", "bastard"])


@pytest.mark.parametrize("msg, n_hits", [
    ("you idiot", 1),
    ("you 1d10t!!!", 1),
    ("fcuk this", 1),               # fuzzy transposition
    ("fcuking hell", 1),            # transposed prefix stem
    ("bastrad", 1),                 # fuzzy transposition on long word
    ("go kill yourself", 2),        # phrase
    ("I love my duck", 0),          # substitution on short word NOT matched
    ("nice shirt", 0),              # insertion on short word NOT matched
    ("hello world", 0),
])
def test_lexicon_find(lex, msg, n_hits):
    assert len(lex.find(msg)) == n_hits


def test_mask_preserves_punctuation_and_spacing(lex):
    masked, hits = lex.mask("You IDIOT!! what the sh1t.")
    assert masked == "You I****!! what the s***."
    assert [h.surface for h in hits] == ["IDIOT", "sh1t"]


def test_mask_full(lex):
    masked, _ = lex.mask("idiot", keep_first=False)
    assert masked == "*****"
