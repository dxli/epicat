"""Locating the leading title card of a clip and reading its episode number."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cjk import find_episode, strip_episode
from .config import TitleConfig
from .ffmpeg import Media, read_frames, read_frames_at
from .imaging import dilate, plate_fill
from .ocr import Ocr, OcrLine
from .util import log


@dataclass
class TitleCard:
    """The leading title card of one clip, if it has one."""
    present: bool
    n_frames: int = 0                  # length of the card, in frames
    ref_frame: int = 0                 # a representative frame index inside the card
    episode: int | None = None
    episode_box: tuple[int, int, int, int] | None = None   # (x, y, w, h) in frame coords
    title_text: str = ""
    lines: list[str] = field(default_factory=list)


def detect_card(media: Media, cfg: TitleConfig) -> TitleCard:
    """Find the run of leading frames that form a static title card.

    The card is identified by its *glyph pattern*: we take a reference frame from
    the first moments of the clip and extend forward while later frames keep
    substantially the same bright pixels. That survives fade-ins and fade-outs,
    which a plain "is this frame dark" test does not.
    """
    if not cfg.enabled:
        return TitleCard(present=False)

    limit = int(cfg.scan_seconds * media.fps_float) + 1
    grays: list[np.ndarray] = []
    for i, frame in enumerate(read_frames(media.path, media.width, media.height,
                                          pix_fmt="gray", duration=cfg.scan_seconds)):
        grays.append(frame.copy())
        if i + 1 >= limit:
            break
    if not grays:
        return TitleCard(present=False)

    masks = [g > cfg.bright_luma for g in grays]
    means = [float(g.mean()) for g in grays]

    ref = -1
    for i in range(min(len(grays), int(2.0 * media.fps_float) + 1)):
        if means[i] <= cfg.dark_max_luma and int(masks[i].sum()) >= cfg.min_bright_px:
            ref = i
            break
    if ref < 0:
        log(f"{Path(media.path).name}: no title card detected", level="debug")
        return TitleCard(present=False)

    # Track the card by how far its glyphs stand out from their immediate
    # surroundings. That decays smoothly through a fade *or* a cross-fade into
    # the first shot, and collapses to near zero once only artwork remains --
    # unlike a raw bright-pixel overlap, which pale artwork keeps triggering.
    ref_mask = masks[ref]
    ring_mask = dilate(ref_mask, 5) & ~dilate(ref_mask, 2)
    if not ring_mask.any():
        ring_mask = ~ref_mask

    def contrast(idx: int) -> float:
        g = grays[idx].astype(np.float32)
        return float(g[ref_mask].mean() - g[ring_mask].mean())

    base = contrast(ref)
    if base < cfg.min_contrast:
        log(f"{Path(media.path).name}: no title card detected (low contrast)", level="debug")
        return TitleCard(present=False)

    end = ref
    max_frames = min(len(grays), int(cfg.max_seconds * media.fps_float))
    while end < max_frames and contrast(end) / base > cfg.end_ratio:
        end += 1

    # Frames before `ref` are the fade-in and belong to the card too.
    card = TitleCard(present=True, n_frames=end, ref_frame=ref)
    log(f"{Path(media.path).name}: title card = frames 0..{end - 1} "
        f"({end / media.fps_float:.2f}s)", level="debug")
    return card


def read_card(media: Media, card: TitleCard, ocr: Ocr, cfg: TitleConfig) -> TitleCard:
    """OCR the card and pull out the episode number plus the box to erase."""
    if not card.present:
        return card
    frames = read_frames_at(media.path, media.width, media.height, [card.ref_frame])
    frame = frames.get(card.ref_frame)
    if frame is None:
        return card

    lines: list[OcrLine] = ocr.read_array(frame)
    card.lines = [ln.text for ln in lines]

    best: tuple[OcrLine, int] | None = None
    for ln in lines:
        hit = find_episode(ln.text)
        if hit is not None:
            # Prefer the most confident match if several lines mention a number.
            if best is None or ln.conf > best[0].conf:
                best = (ln, hit[0])
    if best is not None:
        ln, number = best
        card.episode = number
        card.episode_box = ln.box
        if strip_episode(ln.text):
            # The number shares a line with other words: keep the line, erase
            # only the number's share of it, proportionally.
            card.episode_box = _sub_box(ln, find_episode(ln.text))
    others = [ln.text for ln in lines
              if best is None or ln is not best[0]]
    card.title_text = " ".join(t for t in others if t).strip()
    return card


def _sub_box(line: OcrLine, hit) -> tuple[int, int, int, int]:
    """Approximate the pixel box of a substring inside an OCR'd line.

    Vision reports one box per line; when the episode marker is only part of the
    line we scale by character offsets. Good enough for an erase rectangle.
    """
    number, start, end = hit
    n = max(len(line.text), 1)
    x0 = line.x + int(round(line.w * (start / n)))
    x1 = line.x + int(round(line.w * (end / n)))
    return (x0, line.y, max(x1 - x0, 1), line.h)


def erase_episode_number(frame: np.ndarray, card: TitleCard, cfg: TitleConfig) -> np.ndarray:
    """Paint out the episode-number text on a title frame."""
    if card.episode_box is None:
        return frame
    x, y, w, h = card.episode_box
    m = cfg.erase_margin
    box = (x - m, y - m, w + 2 * m, h + 2 * m)
    return plate_fill(frame, box, ring=cfg.erase_ring, feather_px=cfg.erase_feather)
