"""Area B – Classical NLP: regex normalisation + edit-distance lexicon matcher.

Two public pieces:

* ``normalize_text(text)`` – deterministic cleaner used by *every* model tier.
    "Y0u are a  STUP1D   iiiiidiooot!!! http://x.co"  ->  "you are a stupid idiot"

* ``LexiconMatcher`` – finds offensive tokens in the *original* message
  (exact, prefix, bigram, or Damerau/OSA edit-distance match after
  normalisation) and returns character spans, so the app can mask them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Regex normalisation
# --------------------------------------------------------------------------- #
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "l", "+": "t",
})
_LEET_CHARS = set("0134578@$!|+")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HTML_RE = re.compile(r"<[^>]+>|&[a-z]+;")
_TOKEN_RE = re.compile(r"\S+")
_TRAIL_PUNCT_RE = re.compile(r"^(.*?)([.,!?;:\"')\]]*)$")
_LEAD_PUNCT_RE = re.compile(r"^([\"'(\[]*)(.*)$")
_REPEAT_RE = re.compile(r"([a-z])\1{2,}")          # 3+ identical letters -> 1
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SPACED_RE = re.compile(r"\b(?:[a-z] ){2,}[a-z]\b")  # "f u c k" -> "fuck"
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _deleet_token(token: str) -> str:
    """Convert leetspeak only inside tokens that also contain letters.

    Pure numbers ("2024", "50") are untouched; trailing punctuation such as
    the "!" in "idiot!" is *not* treated as leet.
    """
    lead, rest = _LEAD_PUNCT_RE.match(token).groups()
    core, trail = _TRAIL_PUNCT_RE.match(rest).groups()
    if core and any(c.isalpha() for c in core) and any(c in _LEET_CHARS for c in core):
        core = core.translate(LEET_MAP)
    return lead + core + trail


def deleet(text: str) -> str:
    return _TOKEN_RE.sub(lambda m: _deleet_token(m.group(0)), text)


def reduce_repeats(text: str) -> str:
    """"noooo" -> "no", "fuuuuck" -> "fuck"; double letters ("cool") are kept."""
    return _REPEAT_RE.sub(r"\1", text)


def normalize_text(text: str, join_spaced_letters: bool = True) -> str:
    """Full cleaning pipeline (order matters)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _strip_accents(text).lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _IP_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = deleet(t)                       # before punctuation stripping ("sh!t", "@ss")
    t = reduce_repeats(t)
    t = t.replace("'", "").replace("’", "")   # "don't" -> "dont"
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    if join_spaced_letters:
        t = _SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), t)
    return t


def tokenize(text: str) -> list[str]:
    """Normalise then whitespace-tokenise (used by Word2Vec / lexicon)."""
    return normalize_text(text).split()


# --------------------------------------------------------------------------- #
# Edit-distance lexicon matcher
# --------------------------------------------------------------------------- #
def osa_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Optimal-String-Alignment (restricted Damerau-Levenshtein) distance.

    Counts insertions, deletions, substitutions and adjacent transpositions,
    so "fcuk" -> "fuck" is distance 1.  Early-exits once ``max_dist`` is exceeded.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1
    prev2: list[int] | None = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[lb]


@dataclass(frozen=True)
class Match:
    start: int          # character span in the ORIGINAL message
    end: int
    surface: str        # original text, e.g. "Fcuk"
    term: str           # lexicon entry it matched, e.g. "fuck*"
    rule: str           # exact | prefix | fuzzy | phrase


class LexiconMatcher:
    """Lexicon lookup with conservative fuzzy matching.

    Lexicon file format (one entry per line, ``#`` comments allowed):
        idiot        exact word (after normalisation)
        fuck*        prefix entry: matches fuck, fucking, fucker ...
        kill yourself   two-word phrase

    Fuzzy rule (designed to keep false positives low – e.g. "duck" must not
    match "fuck", "shirt" must not match "shit"):
        * transposition, distance 1, word length >= 4   ("fcuk")
        * insertion/deletion, distance 1, word length >= 5   ("idiiot" -> handled,
          "biatch" style variants)
        * substitution, distance 1, only for long words (>= 7)   ("assholr")
    """

    def __init__(self, terms: Iterable[str]):
        self.exact: set[str] = set()
        self.prefixes: set[str] = set()
        self.phrases: set[tuple[str, ...]] = set()
        for raw in terms:
            t = raw.split("#", 1)[0].strip().lower()
            if not t:
                continue
            if " " in t:
                self.phrases.add(tuple(normalize_text(t).split()))
            elif t.endswith("*"):
                self.prefixes.add(normalize_text(t[:-1]))
            else:
                self.exact.add(normalize_text(t))
        self.exact.discard("")
        self.prefixes.discard("")
        self._fuzzy_pool = sorted(self.exact | self.prefixes)
        self._pool_by_len: dict[int, list[str]] = {}
        for w in self._fuzzy_pool:
            self._pool_by_len.setdefault(len(w), []).append(w)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_files(cls, *paths: Path) -> "LexiconMatcher":
        terms: list[str] = []
        for p in paths:
            p = Path(p)
            if p.exists():
                terms.extend(p.read_text(encoding="utf-8").splitlines())
        return cls(terms)

    @classmethod
    def default(cls) -> "LexiconMatcher":
        from src import config as C
        return cls.from_files(C.SEED_LEXICON_PATH, C.LEARNED_LEXICON_PATH)

    # ------------------------------------------------------------ matching
    @staticmethod
    def _fuzzy_ok(tok: str, word: str) -> bool:
        d = osa_distance(tok, word, max_dist=1)
        if d != 1:
            return False
        if len(tok) == len(word):
            is_transposition = sorted(tok) == sorted(word)
            return (is_transposition and len(word) >= 4) or len(word) >= 7
        return len(word) >= 5  # insertion / deletion

    @lru_cache(maxsize=50_000)
    def match_token(self, norm_tok: str) -> tuple[str, str] | None:
        """Return (term, rule) for a *normalised* token, or None."""
        if not norm_tok or len(norm_tok) < 2:
            return None
        if norm_tok in self.exact:
            return norm_tok, "exact"
        for p in self.prefixes:
            if norm_tok.startswith(p):
                return p + "*", "prefix"
        if len(norm_tok) >= 4:
            # transposed prefix stems: "fcuking" -> fuck*
            for p in self.prefixes:
                if len(p) >= 4 and len(norm_tok) >= len(p):
                    head = norm_tok[:len(p)]
                    if head != p and sorted(head) == sorted(p) and osa_distance(head, p, 1) == 1:
                        return p + "*", "fuzzy"
            n = len(norm_tok)
            candidates = (self._pool_by_len.get(n - 1, []) + self._pool_by_len.get(n, [])
                          + self._pool_by_len.get(n + 1, []))
            for w in candidates:
                if self._fuzzy_ok(norm_tok, w):
                    return (w + "*" if w in self.prefixes else w), "fuzzy"
        return None

    def _spans(self, text: str):
        """Yield (start, end, normalised_token) for each original token's core."""
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            lead, rest = _LEAD_PUNCT_RE.match(tok).groups()
            core, _ = _TRAIL_PUNCT_RE.match(rest).groups()
            norm = normalize_text(core, join_spaced_letters=False).replace(" ", "")
            if norm:
                s = m.start() + len(lead)
                yield s, s + len(core), norm

    def find(self, text: str) -> list[Match]:
        spans = list(self._spans(text))
        hits: dict[int, Match] = {}
        for i, (s, e, norm) in enumerate(spans):
            r = self.match_token(norm)
            if r:
                hits[i] = Match(s, e, text[s:e], r[0], r[1])
        for phrase in self.phrases:
            n = len(phrase)
            for i in range(len(spans) - n + 1):
                if tuple(sp[2] for sp in spans[i:i + n]) == phrase:
                    for k in range(i, i + n):
                        s, e, _ = spans[k]
                        hits.setdefault(k, Match(s, e, text[s:e], " ".join(phrase), "phrase"))
        return [hits[k] for k in sorted(hits)]

    def mask(self, text: str, keep_first: bool = True, char: str = "*") -> tuple[str, list[Match]]:
        """Return (masked_text, matches).  "you fcuking idiot" -> "you f****** i****"."""
        matches = self.find(text)
        out, last = [], 0
        for m in matches:
            out.append(text[last:m.start])
            w = m.surface
            out.append((w[0] + char * (len(w) - 1)) if keep_first and len(w) > 1 else char * len(w))
            last = m.end
        out.append(text[last:])
        return "".join(out), matches


if __name__ == "__main__":
    demos = [
        "Y0u are a  STUP1D   iiiiidiooot!!! http://x.co",
        "what the f.u.c.k is this sh!t",
        "u r such a l0$er, go kill yourself",
        "I love my duck and my new shirt, 50 points!",
        "Don't be an @$$hole",
    ]
    lex = LexiconMatcher.default()
    for d in demos:
        masked, hits = lex.mask(d)
        print(f"RAW : {d}\nNORM: {normalize_text(d)}\nMASK: {masked}  {[h.rule for h in hits]}\n")
