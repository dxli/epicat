"""Dependency-light image helpers (numpy only): PNG I/O, box filters, masking, inpainting."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- PNG


def write_png(path: str | Path, arr: np.ndarray) -> None:
    """Write a uint8 array as PNG. Accepts (h, w) grayscale or (h, w, 3) RGB."""
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    if a.ndim == 2:
        h, w = a.shape
        colour_type = 0
        stride = w
    elif a.ndim == 3 and a.shape[2] == 3:
        h, w, _ = a.shape
        colour_type = 2
        stride = w * 3
    else:
        raise ValueError(f"unsupported array shape for PNG: {a.shape}")

    flat = a.reshape(h, stride)
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type: none
        raw += flat[y].tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


# ------------------------------------------------------------------- filters


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edge-extended. O(n) via a summed-area table."""
    a = np.asarray(a, dtype=np.float32)
    if r <= 0:
        return a
    h, w = a.shape
    pad = np.pad(a, ((r, r), (r, r)), mode="edge")
    sat = np.cumsum(np.cumsum(pad, axis=0, dtype=np.float32), axis=1, dtype=np.float32)
    sat = np.pad(sat, ((1, 0), (1, 0)))
    k = 2 * r + 1
    total = (sat[k:k + h, k:k + w] - sat[0:h, k:k + w]
             - sat[k:k + h, 0:w] + sat[0:h, 0:w])
    return total / float(k * k)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a (2r+1)^2 square."""
    if r <= 0:
        return mask.astype(bool)
    return box_mean(mask.astype(np.float32), r) > 1e-6


def feather(mask: np.ndarray, r: int) -> np.ndarray:
    """Soft alpha in [0, 1]: a dilated mask blurred so the paste edge is not visible."""
    alpha = mask.astype(np.float32)
    if r > 0:
        alpha = np.clip(box_mean(alpha, r) * 1.6, 0.0, 1.0)
        alpha = box_mean(alpha, max(1, r // 2))
    return np.clip(alpha, 0.0, 1.0)


def grey_min(a: np.ndarray, r: int) -> np.ndarray:
    """Sliding-window minimum (grayscale erosion), separable."""
    out = np.asarray(a, dtype=np.float32)
    if r <= 0:
        return out
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = None
        for i in range(2 * r + 1):
            window = [slice(None), slice(None)]
            window[axis] = slice(i, i + n)
            view = padded[tuple(window)]
            acc = view if acc is None else np.minimum(acc, view)
        out = acc
    return out


def grey_max(a: np.ndarray, r: int) -> np.ndarray:
    """Sliding-window maximum (grayscale dilation), separable."""
    return -grey_min(-np.asarray(a, dtype=np.float32), r)


def tophat(a: np.ndarray, r: int) -> np.ndarray:
    """White top-hat: how far each pixel rises above its local background.

    Opening the image with a window wider than a glyph stroke erases the strokes
    and leaves the background; the difference is the strokes alone. This is what
    makes caption detection work when the artwork behind the caption is itself
    near-white.
    """
    a = np.asarray(a, dtype=np.float32)
    return a - grey_max(grey_min(a, r), r)


# --------------------------------------------------------------- text masking


def text_mask(rgb: np.ndarray, *, min_luma: int = 238, max_sat: int = 16,
              min_contrast: float = 0.0, stroke: int = 6) -> np.ndarray:
    """Pixels that look like solid neutral-white caption glyphs.

    `min_luma`/`max_sat` catch bright neutral pixels; `min_contrast` additionally
    requires the pixel to stand above its local background by that much, which
    is what distinguishes a glyph from pale artwork.
    """
    a = rgb.astype(np.float32)
    luma = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    m = (luma > min_luma) & (sat < max_sat)
    if min_contrast > 0:
        m &= tophat(luma, stroke) >= min_contrast
    return m


def grow_mask(rgb: np.ndarray, seed: np.ndarray, *, grow_luma: int = 205,
              grow_sat: int = 45, steps: int = 6, min_contrast: float = 12.0,
              stroke: int = 6) -> np.ndarray:
    """Hysteresis-grow a strict glyph mask over the anti-aliased stroke edges.

    The strict test only catches the solid core of a glyph; its soft edge is
    dimmer and slightly tinted. Growing the seed into connected pixels that pass
    a looser test captures the whole stroke without swallowing the background.
    """
    a = rgb.astype(np.int16)
    luma = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    contrast = tophat(luma, stroke) if min_contrast > 0 else None
    loose = (luma > grow_luma) & (sat < grow_sat)
    if contrast is not None:
        # Without this the growth floods across pale artwork, which passes the
        # brightness test just as a glyph does. Local contrast does not.
        loose &= contrast >= min_contrast
    m = seed.astype(bool).copy()

    # Where the caption sits on near-white artwork the strict seed only catches
    # stroke cores. Inside the seed's own bounding box -- and only there -- a
    # strong local-contrast response is trusted as glyph too.
    if m.any() and min_contrast > 0:
        ys, xs = np.nonzero(m)
        pad = 8
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, m.shape[0])
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, m.shape[1])
        box = np.zeros_like(m)
        box[y0:y1, x0:x1] = True
        strong = box & (luma > grow_luma) & (sat < grow_sat)
        strong &= contrast >= min_contrast * 2.0
        m |= strong

    for _ in range(max(steps, 0)):
        grown = dilate(m, 1) & loose
        grown |= m
        if int(grown.sum()) == int(m.sum()):
            break
        m = grown
    return m


def blend(base: np.ndarray, patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha-weighted composite of `patch` over `base` (both (h, w, 3) uint8)."""
    a3 = alpha[:, :, None].astype(np.float32)
    out = base.astype(np.float32) * (1.0 - a3) + patch.astype(np.float32) * a3
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ inpainting


def _hole_radius(mask: np.ndarray, cap: int = 16) -> int:
    """Largest distance from a hole pixel to the nearest known pixel."""
    known = ~mask
    r = 0
    while r < cap and not dilate(known, r + 1)[mask].all():
        r += 1
    return max(r, 1)


def inpaint(region: np.ndarray, mask: np.ndarray, *, iters: int | None = None,
            smooth: int = 0) -> np.ndarray:
    """Harmonic (Laplace) inpainting of the masked pixels.

    Holes are seeded by a distance-weighted fill from their neighbours, then
    relaxed towards the smooth solution of the Laplace equation. For thin
    strokes over painted artwork this continues the surrounding gradients
    instead of smearing a blurred patch across the area, which is what a plain
    blur-and-blend does.

    Only the mask's bounding box is solved, and the relaxation uses red-black
    successive over-relaxation, which converges in roughly O(r) sweeps where
    plain Jacobi iteration needs O(r^2).
    """
    holes = np.asarray(mask, dtype=bool)
    if not holes.any():
        return region.copy()

    r = _hole_radius(holes)
    ys, xs = np.nonzero(holes)
    pad = r + 2
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, region.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, region.shape[1])

    sub = region[y0:y1, x0:x1]
    sub_holes = holes[y0:y1, x0:x1]
    solved = _relax(sub, sub_holes, r, iters, smooth)

    out = region.copy()
    out[y0:y1, x0:x1] = solved
    return out


def _relax(region: np.ndarray, holes: np.ndarray, r: int,
           iters: int | None, smooth: int) -> np.ndarray:
    u = _seed_fill(region, holes)
    n_iter = iters if iters is not None else int(np.clip(3 * r, 8, 60))

    h, w = holes.shape
    yy, xx = np.indices((h, w))
    parity = (yy + xx) & 1
    red = holes & (parity == 0)
    black = holes & (parity == 1)
    omega = 2.0 / (1.0 + np.sin(np.pi / max(2 * r + 1, 3)))

    for _ in range(n_iter):
        for sel in (red, black):
            pad = np.pad(u, ((1, 1), (1, 1), (0, 0)), mode="edge")
            avg = (pad[:-2, 1:-1] + pad[2:, 1:-1]
                   + pad[1:-1, :-2] + pad[1:-1, 2:]) * 0.25
            u = np.where(sel[:, :, None], u + omega * (avg - u), u)

    if smooth > 0:
        soft = np.dstack([box_mean(u[:, :, c], smooth) for c in range(3)])
        u = np.where(holes[:, :, None], soft, u)
    return np.clip(u + 0.5, 0, 255).astype(np.uint8)


def _seed_fill(region: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Cheap initial guess: distance-weighted average of the nearest known
    pixel above/below and left/right of each hole."""
    out = region.astype(np.float32)
    filled = []
    for axis in (0, 1):
        work = out if axis == 0 else np.transpose(out, (1, 0, 2))
        m = holes if axis == 0 else holes.T
        h = work.shape[0]

        fwd = work.copy()
        fwd_dist = np.full(m.shape, 1e6, dtype=np.float32)
        known = np.zeros(work.shape[1:], dtype=np.float32)
        have = np.zeros(work.shape[1], dtype=bool)
        for y in range(h):
            row_known = ~m[y]
            known[row_known] = work[y][row_known]
            have |= row_known
            fwd[y] = known
            fwd_dist[y] = np.where(row_known, 0.0, (fwd_dist[y - 1] + 1.0) if y else 1e6)
            fwd_dist[y][~have] = 1e6

        bwd = work.copy()
        bwd_dist = np.full(m.shape, 1e6, dtype=np.float32)
        known = np.zeros(work.shape[1:], dtype=np.float32)
        have = np.zeros(work.shape[1], dtype=bool)
        for y in range(h - 1, -1, -1):
            row_known = ~m[y]
            known[row_known] = work[y][row_known]
            have |= row_known
            bwd[y] = known
            bwd_dist[y] = np.where(row_known, 0.0, (bwd_dist[y + 1] + 1.0) if y < h - 1 else 1e6)
            bwd_dist[y][~have] = 1e6

        wf = 1.0 / (fwd_dist + 1.0)
        wb = 1.0 / (bwd_dist + 1.0)
        tot = np.maximum(wf + wb, 1e-6)
        merged = (fwd * wf[:, :, None] + bwd * wb[:, :, None]) / tot[:, :, None]
        merged = np.where(m[:, :, None], merged, work)
        filled.append(merged if axis == 0 else np.transpose(merged, (1, 0, 2)))

    return np.where(holes[:, :, None], (filled[0] + filled[1]) * 0.5, out)


def plate_fill(region: np.ndarray, box: tuple[int, int, int, int], *,
               ring: int = 6, feather_px: int = 4) -> np.ndarray:
    """Erase a rectangle by filling it with the median colour of a ring around it.

    Ideal for text sitting on a flat plate (a black title card, a solid banner).
    `box` is (x, y, w, h) in `region` coordinates.
    """
    x, y, w, h = box
    H, W = region.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return region.copy()

    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(W, x1 + ring), min(H, y1 + ring)
    outer = region[ry0:ry1, rx0:rx1].reshape(-1, 3)
    inner_mask = np.zeros(region.shape[:2], dtype=bool)
    inner_mask[y0:y1, x0:x1] = True
    ring_mask = np.zeros(region.shape[:2], dtype=bool)
    ring_mask[ry0:ry1, rx0:rx1] = True
    ring_mask &= ~inner_mask
    samples = region[ring_mask]
    colour = np.median(samples if len(samples) else outer, axis=0)

    patch = np.broadcast_to(colour.astype(np.float32), region.shape).astype(np.uint8)
    alpha = feather(inner_mask, feather_px)
    return blend(region, patch, alpha)
