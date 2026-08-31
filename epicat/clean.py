"""Per-clip rendering: trim the title card, erase the episode number, erase captions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import overlay
from .bandscan import BandScan, donor_indices
from .config import Config
from .ffmpeg import Media, RawEncoder, read_frames, read_frames_at
from .imaging import blend, dilate, feather, grow_mask, inpaint
from .titles import TitleCard, erase_episode_number
from .util import log

OVERLAY_MARGIN = 16   # px of real surroundings given to inpaint() around a cleared region


@dataclass
class ClipPlan:
    """Everything decided about one clip before a frame is touched."""
    media: Media
    card: TitleCard
    cut_frames: int = 0               # leading frames dropped from the output
    erase_number: bool = True         # paint out the episode number on kept card frames
    scan: BandScan | None = None
    episode: int | None = None
    index: int = 0

    @property
    def cut_seconds(self) -> float:
        return self.cut_frames / self.media.fps_float


@dataclass
class RenderStats:
    frames_written: int = 0
    frames_patched: int = 0
    frames_shifted: int = 0
    frames_inpainted: int = 0
    number_frames: int = 0
    donor_rejected: int = 0
    regions_cleared: int = 0


def _pick_donor(scan: BandScan) -> dict[int, int]:
    """Map every captioned frame to the clean frame it should copy from."""
    choice: dict[int, int] = {}
    for run in scan.runs:
        for i in range(run.start, run.end):
            before, after = run.donor_before, run.donor_after
            if before is None and after is None:
                continue
            if before is None:
                choice[i] = after            # type: ignore[assignment]
            elif after is None:
                choice[i] = before
            else:
                choice[i] = before if (i - before) <= (after - i) else after
    return choice


def _plan_by_frame(scan: BandScan) -> dict[int, list]:
    """Map every frame in a no-donor run to that run's texture-fill plan."""
    out: dict[int, list] = {}
    for run in scan.runs:
        if not run.shift_plan:
            continue
        for i in range(run.start, run.end):
            out[i] = run.shift_plan
    return out


def _apply_shift_plan(band: np.ndarray, mask: np.ndarray, plan: Sequence,
                      bcfg) -> tuple[np.ndarray, np.ndarray]:
    """Apply a run's precomputed texture-fill plan to one frame's band.

    The plan was searched once, on the run's clearest frame; this frame's own
    content may have drifted since (a pan, a slow zoom), so every chunk is
    re-checked here the same way a donor is: does its surrounding ring still
    look like a match? A chunk that no longer does, or a hole pixel the plan
    never covered, is left for the harmonic fallback.

    Returns (band, covered) -- `covered` marks the hole pixels this call
    actually filled; the caller inpaints whatever `mask & ~covered` leaves.
    """
    original = band          # every read comes from here, never from `out`
    out = band.copy()        # every write goes here
    covered = np.zeros_like(mask)
    for chunk in plan:
        y0, y1, x0, x1 = chunk.box
        local_holes = mask[y0:y1, x0:x1]
        if not local_holes.any():
            continue
        H, W = mask.shape
        sy0, sy1 = y0 + chunk.dy, y1 + chunk.dy
        sx0, sx1 = x0 + chunk.dx, x1 + chunk.dx
        if sy0 < 0 or sx0 < 0 or sy1 > H or sx1 > W:
            continue
        ring = dilate(local_holes, bcfg.shift_ring) & ~local_holes
        if ring.any():
            # Adjacent chunks' padded boxes can overlap by a few pixels;
            # reading both sides from `original` keeps this check -- and the
            # patch below -- from ever seeing another chunk's own fill.
            here = original[y0:y1, x0:x1][ring].astype(np.float64)
            there = original[sy0:sy1, sx0:sx1][ring].astype(np.float64)
            # Same metric best_shift() planned against -- MSE normalised by
            # the ring's own variance -- not donor_match_tolerance, which is
            # a raw-difference threshold calibrated for a different scale.
            variance = float(np.var(here)) + 1.0
            quality = float(np.mean((here - there) ** 2)) / variance
            if quality > bcfg.shift_quality_max:
                continue
        patch = original[sy0:sy1, sx0:sx1]
        alpha = feather(local_holes, bcfg.shift_feather)
        out[y0:y1, x0:x1] = blend(original[y0:y1, x0:x1], patch, alpha)
        covered[y0:y1, x0:x1] |= local_holes
    return out, covered


def _overlays(cfg: Config, media: Media) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Work out, once per clip, which pixels each extra region should clear."""
    found = []
    for box in overlay.region_boxes(cfg.band, media):
        mask = overlay.detect(media, box, cfg.band)
        if mask is not None:
            found.append((box, mask))
    return found


def render_clip(plan: ClipPlan, out_path: Path, cfg: Config) -> RenderStats:
    """Write the cleaned video for one clip (video only; audio is handled separately)."""
    media = plan.media
    scan = plan.scan
    stats = RenderStats()
    bcfg = cfg.band

    donors: dict[int, int] = {}
    donor_bands: dict[int, np.ndarray] = {}
    plans: dict[int, list] = {}
    if scan is not None and bcfg.enabled:
        donors = _pick_donor(scan)
        wanted = donor_indices(scan)
        if wanted:
            donor_bands = read_frames_at(
                media.path, media.width, scan.band.height, wanted,
                vf=scan.band.crop_filter(media.width))
            log(f"clip {plan.index}: fetched {len(donor_bands)}/{len(wanted)} donor frames",
                level="debug")
        plans = _plan_by_frame(scan)

    extra = _overlays(cfg, media)
    y0 = scan.band.y0 if scan else 0
    y1 = scan.band.y1 if scan else 0

    enc = RawEncoder(out_path, media.width, media.height, media.fps,
                     vcodec=cfg.video.codec, crf=cfg.video.crf,
                     preset=cfg.video.preset, pix_fmt=cfg.video.pix_fmt)
    try:
        for idx, frame in enumerate(read_frames(media.path, media.width, media.height)):
            if idx < plan.cut_frames:
                continue
            out = frame

            if plan.erase_number and plan.card.present and idx < plan.card.n_frames:
                out = erase_episode_number(out, plan.card, cfg.title)
                stats.number_frames += 1

            if scan is not None and bcfg.enabled and idx < scan.n_frames and scan.has_text[idx]:
                band = out[y0:y1]
                mask = dilate(grow_mask(band, scan.mask(idx),
                                        grow_luma=bcfg.grow_luma,
                                        grow_sat=bcfg.grow_sat,
                                        steps=bcfg.grow_steps,
                                        min_contrast=bcfg.grow_contrast,
                                        stroke=bcfg.stroke), bcfg.dilate)
                donor_idx = donors.get(idx)
                patch = donor_bands.get(donor_idx) if donor_idx is not None else None
                if patch is not None and not _donor_matches(band, patch, mask, bcfg):
                    patch = None
                    stats.donor_rejected += 1
                if patch is not None:
                    alpha = feather(mask, bcfg.feather)
                    band = blend(band, patch, alpha)
                    stats.frames_patched += 1
                else:
                    frame_plan = plans.get(idx)
                    remaining = mask
                    if frame_plan:
                        band, covered = _apply_shift_plan(band, mask, frame_plan, bcfg)
                        if covered.any():
                            stats.frames_shifted += 1
                        remaining = mask & ~covered
                    if remaining.any():
                        band = inpaint(band, remaining, smooth=max(bcfg.dilate, 2))
                        stats.frames_inpainted += 1
                out = out.copy() if out is frame else out
                out[y0:y1] = band

            for (bx, by, bw, bh), rmask in extra:
                if bw <= 0 or bh <= 0:
                    continue
                # Give inpaint() a margin of real surroundings to draw from --
                # without it, a mask that fills the whole box (overlay
                # detection's own "just treat all of it as overlay" fallback)
                # would leave nothing known anywhere nearby to interpolate.
                m = OVERLAY_MARGIN
                ry0, ry1 = max(by - m, 0), min(by + bh + m, out.shape[0])
                rx0, rx1 = max(bx - m, 0), min(bx + bw + m, out.shape[1])
                region = out[ry0:ry1, rx0:rx1]
                if region.size == 0:
                    continue
                full_mask = np.zeros(region.shape[:2], dtype=bool)
                full_mask[by - ry0:by - ry0 + bh, bx - rx0:bx - rx0 + bw] = rmask
                out = out.copy() if out is frame else out
                out[ry0:ry1, rx0:rx1] = inpaint(region, full_mask)
                stats.regions_cleared += 1

            enc.write(out)
            stats.frames_written += 1
    except BaseException:
        # Anything short of every frame being written -- Ctrl-C, a decode
        # error, a dead encoder -- must not publish a truncated segment under
        # the name a resumed run would trust as finished.
        enc.abort()
        raise
    else:
        enc.close()
    return stats


def _donor_matches(band: np.ndarray, donor: np.ndarray, mask: np.ndarray,
                   bcfg) -> bool:
    """Reject a donor whose background no longer lines up with this frame.

    Compares the two around the glyphs but outside them. Cheap insurance
    against a missed shot change producing a visibly wrong patch.
    """
    ring = dilate(mask, 8) & ~mask
    if not ring.any():
        return True
    a = band[ring].astype(np.float32)
    b = donor[ring].astype(np.float32)
    return float(np.abs(a - b).mean()) <= bcfg.donor_match_tolerance
