"""Parsing episode numbers out of OCR'd title text (Chinese numerals and Latin forms)."""
from __future__ import annotations

import re
from typing import Optional

_DIGITS = {
    "〇": 0, "零": 0, "○": 0, "０": 0,
    "一": 1, "壹": 1, "幺": 1,
    "二": 2, "贰": 2, "两": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}

# Full-width digits map onto ASCII for the numeric branch.
_FULLWIDTH = {chr(0xFF10 + i): str(i) for i in range(10)}


def chinese_to_int(text: str) -> Optional[int]:
    """Convert a Chinese numeral string (up to 9999) to an int. Returns None if unparseable."""
    s = "".join(_FULLWIDTH.get(c, c) for c in text).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    total = 0
    section = 0
    seen = False
    for ch in s:
        if ch in _DIGITS:
            section = _DIGITS[ch]
            seen = True
        elif ch in _UNITS:
            unit = _UNITS[ch]
            if section == 0:
                section = 1  # bare 十 == 10
            if unit >= 100:
                total += section * unit
                section = 0
            else:
                total += section * unit
                section = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return total + section


# Ordered by specificity. Each pattern must expose the number as group "n".
EPISODE_PATTERNS = (
    r"第\s*(?P<n>[0-9０-９〇零○一壹二贰两三叁四肆五伍六陆七柒八捌九玖十拾百佰千仟]+)\s*[集话話期回]",
    r"(?:^|\b)(?:EP|Ep|ep)\.?\s*(?P<n>\d{1,4})(?:\b|$)",
    r"(?:^|\b)(?:Episode|EPISODE|episode)\s*(?P<n>\d{1,4})(?:\b|$)",
    r"(?:^|\b)(?:Part|PART|part)\s*(?P<n>\d{1,4})(?:\b|$)",
)


def find_episode(text: str, patterns: tuple[str, ...] = EPISODE_PATTERNS) -> Optional[tuple[int, int, int]]:
    """Find an episode marker in `text`.

    Returns (episode_number, match_start, match_end) or None.
    """
    for pat in patterns:
        for m in re.finditer(pat, text):
            n = chinese_to_int(m.group("n"))
            if n is not None and 0 <= n <= 9999:
                return n, m.start(), m.end()
    return None


def strip_episode(text: str, patterns: tuple[str, ...] = EPISODE_PATTERNS) -> str:
    """Remove every episode marker from `text` (used when rebuilding a title string)."""
    out = text
    for pat in patterns:
        out = re.sub(pat, "", out)
    return re.sub(r"\s{2,}", " ", out).strip(" -–—·:：|")
