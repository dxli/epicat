"""Finding the burnt-in caption band, and mapping out where captions appear."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import SubtitleBandConfig
from .ffmpeg import Media, read_frames
from .imaging import text_mask
from .util import log


@dataclass
class Band:
    y0: int
    y1: int                  # exclusive

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop_filter(self, width: int) -> str:
        return f"crop={width}:{self.height}:0:{self.y0}"


@dataclass
class Run:
    """A stretch of consecutive frames showing one caption."""
    start: int
    end: int                 # exclusive
    donor_before: int | None = None
    donor_after: int | None = None

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class BandScan:
    band: Band
    width: int
    n_frames: int
    counts: np.ndarray                  # glyph pixels per frame
    has_text: np.ndarray                # bool per frame
    packed: list[np.ndarray]            # per-frame glyph mask, np.packbits
    shot_break: np.ndarray              # bool per frame: band content jumped here
    runs: list[Run] = field(default_factory=list)

    def mask(self, i: int) -> np.ndarray:
        flat = np.unpackbits(self.packed[i], count=self.band.height * self.width)
        return flat.reshape(self.band.height, self.width).astype(bool)


def _row_histogram(media: Media, cfg: SubtitleBandConfig, samples: int) -> tuple[np.ndarray, int]:
    """How often each row holds glyph-like pixels, over frames sampled evenly."""
    step = max(media.duration / max(samples, 1), 0.25)
    rows = np.zeros(media.height, dtype=np.float32)
    seen = 0
    for frame in read_frames(media.path, media.width, media.height,
                             vf=f"fps=1/{step:.4f}"):
        m = text_mask(frame, min_luma=cfg.min_luma, max_sat=cfg.max_sat)
        rows += (m.sum(axis=1) >= 3).astype(np.float32)
        seen += 1
        if seen >= samples * 2:
            break
    return rows, seen


def auto_band(media: Media, cfg: SubtitleBandConfig, *,
              samples: int | None = None) -> Band | None:
    """Locate the caption band in a single clip."""
    return auto_band_multi([media], cfg, samples=samples)


def auto_band_multi(medias: Sequence[Media], cfg: SubtitleBandConfig, *,
                    samples: int | None = None) -> Band | None:
    """Locate the caption band from the distribution of glyph-like pixels.

    Captions are solid neutral white; artwork almost never is. Accumulating a
    row histogram of such pixels makes the band obvious without needing OCR.
    Sampling every clip rather than just the first keeps one atypical opening
    from deciding the geometry for the whole series.
    """
    if not medias:
        return None
    ref = medias[0]
    if cfg.top is not None and cfg.bottom is not None:
        return Band(int(cfg.top * ref.height), int(cfg.bottom * ref.height))

    per_clip = max((samples or cfg.auto_samples) // max(len(medias), 1), 6)
    rows = np.zeros(ref.height, dtype=np.float32)
    seen = 0
    for media in medias:
        if media.height != ref.height:
            continue
        hist, n = _row_histogram(media, cfg, per_clip)
        rows += hist
        seen += n
    if seen == 0 or rows.max() < 2:
        log("no caption-like text found; caption erasure will be skipped", level="warn")
        return None

    floor = int(cfg.search_top * ref.height)
    limited = rows.copy()
    limited[:floor] = 0
    if limited.max() < 2:
        limited = rows

    thresh = limited.max() * 0.15
    hot = limited >= thresh
    best = (0, 0, 0)
    y = 0
    while y < ref.height:
        if hot[y]:
            s0 = y
            while y < ref.height and hot[y]:
                y += 1
            if y - s0 > best[0]:
                best = (y - s0, s0, y)
        else:
            y += 1
    if best[0] == 0:
        return None

    y0 = max(0, best[1] - cfg.pad)
    y1 = min(ref.height, best[2] + cfg.pad)
    log(f"caption band y={y0}..{y1} ({y1 - y0} px, "
        f"{y0 / ref.height:.3f}..{y1 / ref.height:.3f}) "
        f"from {seen} sampled frames across {len(medias)} clip(s)")
    return Band(y0, y1)


def _block_means(strip: np.ndarray, block: int = 8) -> np.ndarray:
    """Coarse (h/block, w/block, 3) signature of a band strip."""
    h, w = strip.shape[:2]
    bh, bw = max(h // block, 1), max(w // block, 1)
    cropped = strip[:bh * block, :bw * block].astype(np.float32)
    return cropped.reshape(bh, block, bw, block, 3).mean(axis=(1, 3))


def scan(media: Media, band: Band, cfg: SubtitleBandConfig) -> BandScan:
    """One cheap pass decoding only the band: glyph masks, counts, shot breaks."""
    counts: list[int] = []
    packed: list[np.ndarray] = []
    sigs: list[np.ndarray] = []
    for strip in read_frames(media.path, media.width, band.height,
                             vf=band.crop_filter(media.width)):
        m = text_mask(strip, min_luma=cfg.min_luma, max_sat=cfg.max_sat)
        counts.append(int(m.sum()))
        packed.append(np.packbits(m.ravel()))
        sigs.append(_block_means(strip))

    n = len(counts)
    counts_a = np.asarray(counts, dtype=np.int32)
    has_text = counts_a >= cfg.min_px

    # Shot changes: compare coarse block signatures and take the *median* block
    # difference. A cut moves every block; a caption appearing moves only the
    # few blocks it covers, so the median ignores it.
    breaks = np.zeros(n, dtype=bool)
    if n > 1:
        sig = np.stack(sigs)
        diff = np.abs(np.diff(sig, axis=0)).sum(axis=3)
        delta = np.median(diff.reshape(n - 1, -1), axis=1)
        breaks[1:] = delta > cfg.shot_break_delta

    sc = BandScan(band=band, width=media.width, n_frames=n, counts=counts_a,
                  has_text=has_text, packed=packed, shot_break=breaks)
    sc.runs = find_runs(has_text, cfg)
    assign_donors(sc, cfg)
    return sc


def find_runs(has_text: np.ndarray, cfg: SubtitleBandConfig) -> list[Run]:
    """Group captioned frames into runs, bridging brief drop-outs."""
    n = len(has_text)
    runs: list[Run] = []
    i = 0
    while i < n:
        if not has_text[i]:
            i += 1
            continue
        s = i
        gap = 0
        j = i
        last_true = i
        while j < n:
            if has_text[j]:
                last_true = j
                gap = 0
            else:
                gap += 1
                if gap > cfg.merge_gap_frames:
                    break
            j += 1
        runs.append(Run(start=s, end=last_true + 1))
        i = last_true + 1
    return [r for r in runs if len(r) >= cfg.min_run_frames]


def assign_donors(sc: BandScan, cfg: SubtitleBandConfig) -> None:
    """For each caption run, pick clean frames to copy the background from.

    Only frames from the same shot qualify: pasting background across a cut
    would be worse than the caption. Runs that cover a whole shot get no donor
    and are repaired by inpainting instead.
    """
    n = sc.n_frames
    limit = cfg.donor_search_frames

    for run in sc.runs:
        run.donor_before = None
        run.donor_after = None

        i = run.start - 1
        while i >= 0 and run.start - i <= limit:
            if sc.shot_break[i + 1]:
                break
            if not sc.has_text[i]:
                run.donor_before = i
                break
            i -= 1

        j = run.end
        while j < n and j - run.end <= limit:
            if sc.shot_break[j]:
                break
            if not sc.has_text[j]:
                run.donor_after = j
                break
            j += 1


def donor_indices(scan_: BandScan) -> list[int]:
    out: set[int] = set()
    for r in scan_.runs:
        if r.donor_before is not None:
            out.add(r.donor_before)
        if r.donor_after is not None:
            out.add(r.donor_after)
    return sorted(out)
