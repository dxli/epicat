"""OCR backends. macOS Vision (via the bundled `macocr` helper) is preferred;
Tesseract is a portable fallback."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .imaging import write_png
from .util import ToolError, log, run_text

_HERE = Path(__file__).resolve().parent
_TOOL_SRC = _HERE.parent / "tools" / "macocr" / "macocr.swift"
_TOOL_BIN = _HERE.parent / "tools" / "macocr" / "macocr"


@dataclass
class OcrLine:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


class OcrBackend:
    name = "none"

    def available(self) -> bool:
        return False

    def read(self, png_path: Path, langs: Sequence[str]) -> list[OcrLine]:
        raise NotImplementedError


class VisionOcr(OcrBackend):
    """Apple Vision text recognition. Offline, and strong on Simplified Chinese."""

    name = "vision"

    def __init__(self) -> None:
        self._checked = False

    def _ensure_built(self) -> bool:
        if _TOOL_BIN.exists():
            return True
        if not _TOOL_SRC.exists() or not shutil.which("swiftc"):
            return False
        log("building macocr helper (one-off)…")
        proc = subprocess.run(["swiftc", "-O", "-o", str(_TOOL_BIN), str(_TOOL_SRC)],
                              capture_output=True)
        if proc.returncode != 0:
            log(f"macocr build failed: {proc.stderr.decode('utf-8', 'replace')[:400]}", level="warn")
            return False
        return _TOOL_BIN.exists()

    def available(self) -> bool:
        if not self._checked:
            self._ok = self._ensure_built()
            self._checked = True
        return self._ok

    def read(self, png_path: Path, langs: Sequence[str]) -> list[OcrLine]:
        out = run_text([str(_TOOL_BIN), str(png_path), "--langs", ",".join(langs), "--min-conf", "0.0"])
        data = json.loads(out)
        return [OcrLine(**ln) for ln in data["lines"]]


class TesseractOcr(OcrBackend):
    name = "tesseract"

    _LANG_MAP = {"zh-Hans": "chi_sim", "zh-Hant": "chi_tra", "en-US": "eng", "en": "eng", "ja": "jpn"}

    def available(self) -> bool:
        return shutil.which("tesseract") is not None

    def read(self, png_path: Path, langs: Sequence[str]) -> list[OcrLine]:
        installed = set(run_text(["tesseract", "--list-langs"]).split())
        want = [self._LANG_MAP.get(l, l) for l in langs]
        use = [l for l in want if l in installed] or ["eng"]
        tsv = run_text(["tesseract", str(png_path), "stdout", "-l", "+".join(use), "--psm", "6", "tsv"])
        rows = [r.split("\t") for r in tsv.splitlines()[1:] if r.strip()]
        buckets: dict[tuple, list] = {}
        for r in rows:
            if len(r) < 12 or not r[11].strip():
                continue
            key = tuple(r[1:5])  # page/block/par/line
            try:
                conf = float(r[10])
            except ValueError:
                continue
            buckets.setdefault(key, []).append((int(r[6]), int(r[7]), int(r[8]), int(r[9]), conf, r[11]))
        lines: list[OcrLine] = []
        for words in buckets.values():
            xs = [w[0] for w in words]
            ys = [w[1] for w in words]
            x1 = max(w[0] + w[2] for w in words)
            y1 = max(w[1] + w[3] for w in words)
            joiner = "" if any(ord(c) > 0x2E80 for w in words for c in w[5]) else " "
            lines.append(OcrLine(
                text=joiner.join(w[5] for w in words),
                conf=float(np.mean([w[4] for w in words])) / 100.0,
                x=min(xs), y=min(ys), w=x1 - min(xs), h=y1 - min(ys),
            ))
        return lines


_BACKENDS = {"vision": VisionOcr, "tesseract": TesseractOcr}


class Ocr:
    """Front door: picks a backend and OCRs numpy arrays or files."""

    def __init__(self, backend: str = "auto", langs: Sequence[str] = ("zh-Hans", "en-US")):
        self.langs = list(langs)
        candidates = list(_BACKENDS) if backend == "auto" else [backend]
        self.backend: OcrBackend | None = None
        for name in candidates:
            cls = _BACKENDS.get(name)
            if cls is None:
                raise ToolError(f"unknown OCR backend: {name}")
            inst = cls()
            if inst.available():
                self.backend = inst
                break
        if self.backend is None:
            raise ToolError(
                "no OCR backend available. On macOS this needs `swiftc` (Xcode command line "
                "tools) for the Vision backend, or install tesseract with a Chinese language pack."
            )
        log(f"OCR backend: {self.backend.name}")

    def read_file(self, path: str | Path) -> list[OcrLine]:
        assert self.backend
        return self.backend.read(Path(path), self.langs)

    def read_array(self, arr: np.ndarray, *, upscale: int = 1,
                   gamma: float = 1.0) -> list[OcrLine]:
        """OCR a numpy image.

        `upscale` repeats pixels first, which materially improves recognition of
        small caption text. `gamma` above 1 darkens the midtones while leaving
        white alone, which is what makes white captions readable when they sit
        on near-white artwork.
        """
        img = arr
        if gamma and abs(gamma - 1.0) > 1e-3:
            img = np.clip(255.0 * (img.astype(np.float32) / 255.0) ** gamma + 0.5,
                          0, 255).astype(np.uint8)
        if upscale > 1:
            img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            tmp = Path(fh.name)
        try:
            write_png(tmp, img)
            lines = self.read_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        if upscale > 1:
            for ln in lines:
                ln.x //= upscale
                ln.y //= upscale
                ln.w //= upscale
                ln.h //= upscale
        return lines
