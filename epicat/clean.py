"""Per-clip rendering: trim the title card, erase the episode number, erase captions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    if scan is not None and bcfg.enabled:
        donors = _pick_donor(scan)
        wanted = donor_indices(scan)
        if wanted:
            donor_bands = read_frames_at(
                media.path, media.width, scan.band.height, wanted,
                vf=scan.band.crop_filter(media.width))
            log(f"clip {plan.index}: fetched {len(donor_bands)}/{len(wanted)} donor frames",
                level="debug")

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
                    band = inpaint(band, mask, smooth=max(bcfg.dilate, 2))
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
