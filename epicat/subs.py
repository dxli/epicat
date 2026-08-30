"""Subtitle cues plus SRT/VTT reading and writing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .util import atomic_output


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    def shifted(self, offset: float) -> "Cue":
        return Cue(self.start + offset, self.end + offset, self.text)


def _stamp(t: float, sep: str = ",") -> str:
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(path: str | Path, cues: Sequence[Cue]) -> None:
    parts = []
    for i, c in enumerate(cues, 1):
        parts.append(f"{i}\n{_stamp(c.start)} --> {_stamp(c.end)}\n{c.text}\n")
    with atomic_output(path) as tmp:
        tmp.write_text("\n".join(parts), encoding="utf-8")


def write_vtt(path: str | Path, cues: Sequence[Cue]) -> None:
    parts = ["WEBVTT\n"]
    for c in cues:
        parts.append(f"{_stamp(c.start, '.')} --> {_stamp(c.end, '.')}\n{c.text}\n")
    with atomic_output(path) as tmp:
        tmp.write_text("\n".join(parts), encoding="utf-8")


_TS = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _parse_stamp(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def read_srt(path: str | Path) -> list[Cue]:
    text = Path(path).read_text(encoding="utf-8-sig")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        # maxsplit=1: a hand-edited file (this is a documented resume
        # workflow) might leave stray text after the timestamps on that line.
        left, _, right = lines[0].partition("-->")
        cues.append(Cue(_parse_stamp(left), _parse_stamp(right), "\n".join(lines[1:]).strip()))
    return cues


def shift_all(cues: Iterable[Cue], offset: float) -> list[Cue]:
    return [c.shifted(offset) for c in cues]


def merge(groups: Iterable[Sequence[Cue]]) -> list[Cue]:
    out: list[Cue] = []
    for g in groups:
        out.extend(g)
    out.sort(key=lambda c: c.start)
    return out


def bilingual(primary: Sequence[Cue], secondary: Sequence[Cue]) -> list[Cue]:
    """Stack two aligned tracks into one (same count and timing assumed)."""
    out = []
    for i, c in enumerate(primary):
        extra = secondary[i].text if i < len(secondary) else ""
        out.append(Cue(c.start, c.end, f"{c.text}\n{extra}".strip()))
    return out
