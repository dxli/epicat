"""Reading the burnt-in captions as text, before they are erased.

The captions are the show's own script, so OCR'ing them yields better source
text — and frame-exact timing — than transcribing the audio.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .bandscan import BandScan
from .config import SubtitleBandConfig, TextConfig
from .ffmpeg import Media, read_frames_at
from .ocr import Ocr
from .subs import Cue
from .util import log

_WS = re.compile(r"[\s　]+")


def normalise(text: str) -> str:
    t = _WS.sub("", text)
    return t.strip(" .,:;-—·")


def similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass
class Sample:
    frame: int
    text: str
    conf: float
    weight: float = 1.0      # how completely the caption was drawn on this frame


def containment(a: str, b: str) -> float:
    """Share of the shorter string's characters that the longer one also has."""
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    pool = Counter(long)
    hit = 0
    for ch in short:
        if pool[ch] > 0:
            pool[ch] -= 1
            hit += 1
    return hit / len(short)


def same_caption(a: str, b: str, threshold: float) -> bool:
    """Two OCR reads belong to the same caption.

    Captions here fade or wipe in, so early frames yield a fragment of the final
    line. A fragment is not similar to the whole under an edit-distance ratio,
    but it is contained in it, which is the test that matters.
    """
    if similar(a, b) >= threshold:
        return True
    return len(min(a, b, key=len)) >= 2 and containment(a, b) >= 0.8


def _sample_frames(scan: BandScan, fps: float, sample_hz: float) -> list[int]:
    step = max(int(round(fps / max(sample_hz, 0.5))), 1)
    frames: list[int] = []
    for run in scan.runs:
        idx = list(range(run.start + step // 2, run.end, step))
        if not idx:
            idx = [(run.start + run.end) // 2]
        frames.extend(i for i in idx if i < scan.n_frames)
    return sorted(set(frames))


def extract(media: Media, scan: BandScan, ocr: Ocr, tcfg: TextConfig,
            bcfg: SubtitleBandConfig) -> list[Cue]:
    """OCR the caption band at intervals and fold the samples into timed cues."""
    fps = media.fps_float
    frames = _sample_frames(scan, fps, tcfg.ocr_sample_hz)
    if not frames:
        return []
    log(f"OCR sampling {len(frames)} caption frames…")

    strips = read_frames_at(media.path, media.width, scan.band.height, frames,
                            vf=scan.band.crop_filter(media.width))

    samples: list[Sample] = []
    for f in frames:
        strip = strips.get(f)
        if strip is None:
            continue
        lines = ocr.read_array(strip, upscale=tcfg.ocr_upscale,
                               gamma=tcfg.ocr_gamma)
        lines = [ln for ln in lines if normalise(ln.text)]
        if not lines:
            continue
        lines.sort(key=lambda ln: (ln.y, ln.x))
        text = normalise("".join(ln.text for ln in lines))
        conf = float(np.mean([ln.conf for ln in lines]))
        samples.append(Sample(frame=f, text=text, conf=conf,
                              weight=float(scan.counts[f])))

    samples = _drop_watermarks(samples)
    if not samples:
        return []

    groups = _group(samples, tcfg.caption_similarity, fps)
    cues = tidy(_to_cues(groups, scan, fps, bcfg), tcfg)
    log(f"recovered {len(cues)} captions from {len(samples)} OCR samples")
    return cues


def tidy(cues: list[Cue], tcfg: TextConfig) -> list[Cue]:
    """Drop leftovers from the fade-in problem and merge repeats.

    Grouping catches most fragments, but a caption that fades in slowly can
    still leave a brief cue holding a few characters of its neighbour. Such a
    cue is short *and* its text is contained in the line next to it, which the
    real lines never are.
    """
    kept: list[Cue] = []
    for i, cue in enumerate(cues):
        text = cue.text
        if cue.duration <= tcfg.fragment_max_seconds:
            neighbours = [cues[j].text for j in (i - 1, i + 1) if 0 <= j < len(cues)]
            if any(n != text and containment(text, n) >= 0.95 and len(n) > len(text)
                   for n in neighbours):
                continue
        if kept and kept[-1].text == text and cue.start - kept[-1].end < 0.5:
            kept[-1].end = cue.end       # one caption that briefly dipped out
            continue
        kept.append(cue)
    return kept


def _drop_watermarks(samples: list[Sample], threshold: float = 0.55) -> list[Sample]:
    """Discard text that is present almost all the time — that is a watermark."""
    if len(samples) < 8:
        return samples
    counts = Counter(s.text for s in samples)
    common = {t for t, c in counts.items() if c / len(samples) >= threshold}
    if not common:
        return samples
    log(f"ignoring persistent overlay text: {sorted(common)!r}", level="debug")
    return [s for s in samples if s.text not in common]


def _group(samples: list[Sample], threshold: float, fps: float,
           max_span: float = 12.0) -> list[list[Sample]]:
    groups: list[list[Sample]] = []
    for s in samples:
        if groups:
            cur = groups[-1]
            span = (s.frame - cur[0].frame) / fps
            best = max(cur, key=lambda x: x.weight)
            if span <= max_span and (same_caption(cur[-1].text, s.text, threshold)
                                     or same_caption(best.text, s.text, threshold)):
                cur.append(s)
                continue
        groups.append([s])
    return groups


def _best_text(group: list[Sample]) -> str:
    """Vote across the samples of one caption.

    Each sample votes with a weight that rises steeply with how many glyph
    pixels were on screen, so frames from the middle of a fade barely count and
    the fully drawn line wins. Only well-drawn frames may supply a candidate,
    which keeps a fragment from winning just by being frequent.
    """
    peak = max((s.weight for s in group), default=0.0) or 1.0
    weights = {id(s): s.conf * (s.weight / peak) ** 3 for s in group}
    candidates = {s.text for s in group if s.weight >= 0.75 * peak and s.text}
    if not candidates:
        candidates = {s.text for s in group if s.text}
    if not candidates:
        return ""
    return max(candidates,
               key=lambda t: (sum(weights[id(s)] * similar(t, s.text) for s in group), len(t)))


def _to_cues(groups: list[list[Sample]], scan: BandScan, fps: float,
             bcfg: SubtitleBandConfig) -> list[Cue]:
    cues: list[Cue] = []
    n = scan.n_frames
    for gi, group in enumerate(groups):
        text = _best_text(group)
        if not text:
            continue
        first, last = group[0].frame, group[-1].frame

        # Extend outwards over frames that are still captioned …
        start = first
        while start - 1 >= 0 and scan.has_text[start - 1]:
            start -= 1
        end = last
        while end + 1 < n and scan.has_text[end + 1]:
            end += 1

        # … but never past a neighbouring caption: split the difference.
        if gi > 0:
            prev_last = groups[gi - 1][-1].frame
            if start <= prev_last:
                start = (prev_last + first) // 2 + 1
        if gi + 1 < len(groups):
            next_first = groups[gi + 1][0].frame
            if end >= next_first:
                end = (last + next_first) // 2

        if end < start:
            end = start
        cues.append(Cue(start=start / fps, end=(end + 1) / fps, text=text))

    # Clamp any residual overlap and drop degenerate cues.
    out: list[Cue] = []
    for cue in cues:
        if out and cue.start < out[-1].end:
            mid = (out[-1].end + cue.start) / 2
            out[-1].end = mid
            cue.start = mid
        if cue.end - cue.start >= 0.15:
            out.append(cue)
    return out
