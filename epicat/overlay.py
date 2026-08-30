"""Finding static overlays — a watermark, a channel bug — inside a fixed region.

Unlike a caption, a station overlay is in every frame at the same place and is
often semi-transparent, so no single frame shows it clearly enough to threshold.
It does, however, hold still while the artwork behind it moves: averaging each
pixel's local-contrast response over many frames leaves the overlay standing and
averages the artwork away.
"""
from __future__ import annotations

import numpy as np

from .config import SubtitleBandConfig
from .ffmpeg import Media, read_frames
from .imaging import dilate, tophat
from .util import log


def region_boxes(cfg: SubtitleBandConfig, media: Media) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r in cfg.extra_regions:
        if len(r) != 4:
            raise ValueError(f"extra_regions entries need [x, y, w, h], got {r!r}")
        x, y, w, h = r
        boxes.append((int(x * media.width), int(y * media.height),
                      int(w * media.width), int(h * media.height)))
    return boxes


def detect(media: Media, box: tuple[int, int, int, int],
           cfg: SubtitleBandConfig, *, samples: int = 48) -> np.ndarray | None:
    """Return a mask of the overlay inside `box`, or None if nothing persists."""
    bx, by, bw, bh = box
    if bw <= 0 or bh <= 0:
        return None
    step = max(media.duration / max(samples, 1), 0.2)
    vf = f"crop={bw}:{bh}:{bx}:{by},fps=1/{step:.4f}"

    total = np.zeros((bh, bw), dtype=np.float32)
    seen = 0
    for strip in read_frames(media.path, bw, bh, vf=vf):
        luma = strip.astype(np.float32).mean(axis=2)
        total += np.abs(tophat(luma, cfg.stroke))
        seen += 1
        if seen >= samples:
            break
    if seen < 4:
        return None

    mean = total / seen
    # A pixel belongs to the overlay when its response stays well above the
    # region's own background level, frame after frame.
    floor = float(np.percentile(mean, 55))
    spread = float(np.percentile(mean, 97) - floor)
    if spread < cfg.extra_contrast * 0.4:
        log(f"no persistent overlay found in region {box}", level="warn")
        return None

    mask = mean >= floor + spread * cfg.extra_persistence
    mask = dilate(mask, cfg.dilate)
    coverage = float(mask.mean())
    if coverage > 0.75:
        log(f"overlay detection in region {box} matched {coverage:.0%} of the box; "
            "treating the whole rectangle as overlay", level="warn")
        mask = np.ones_like(mask)
    log(f"overlay in region {box}: {int(mask.sum())} px "
        f"({coverage:.0%} of the box) from {seen} frames", level="debug")
    return mask
