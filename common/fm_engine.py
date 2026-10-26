"""
FM-index engine backing the alignment, via the `fm-index` PyPI library (a real
BWT FM-index; `pip install fm-index`).

The C++ maintains an SA interval and extends it one symbol to the right; its width
is the number of occurrences of the matched pattern. This library does not expose
an interval handle, but it exposes `count(pattern)` whose cost depends only on the
pattern length, not on the number of occurrences. So we represent the "interval"
by the matched pattern string P, and:

    forward_search(P, c) feasible  <=>  count(P + c) > 0
    interval width                  =   count(P)

which is exactly the C++'s right-extension of the SA interval. Because count() is
~1 microsecond regardless of how many occurrences P has, this is orders of
magnitude faster than scanning an explicit position list, and makes full-log runs
feasible in pure Python.

Assumes single-character labels (true for the BD/SH datasets: A..O / A..K plus the
',' separator), so the concatenated model text is an ordinary byte string.
"""

from __future__ import annotations

from fm_index import FMIndex


class FMEngine:
    def __init__(self, text: str):
        self.text = text
        self.fm = FMIndex(text)

    def feasible(self, pattern: str, c: str) -> bool:
        return self.fm.count(pattern + c) > 0

    def count(self, pattern: str) -> int:
        if pattern == "":
            return len(self.text)
        return self.fm.count(pattern)
