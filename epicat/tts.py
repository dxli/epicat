"""Text-to-speech backends for building a dubbed audio track."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import TextConfig
from .util import ToolError, log, run

_HERE = Path(__file__).resolve().parent
KOKORO_DIR = _HERE.parent / "tools" / "kokoro"

# The Kokoro model is a few hundred MB; reuse a cache the user already has.
KOKORO_CACHE_CANDIDATES = (
    KOKORO_DIR / "models",
    Path.home() / ".cache" / "epicat" / "kokoro",
)


@dataclass
class Utterance:
    id: str
    text: str
    speed: float = 1.0


class TtsBackend:
    name = "none"
    default_voice = ""

    def available(self) -> bool:
        return False

    def synth(self, items: Sequence[Utterance], out_dir: Path, voice: str) -> dict[str, Path]:
        raise NotImplementedError


class KokoroTts(TtsBackend):
    """kokoro-js running under node. Noticeably more natural than `say`, and it
    takes a speaking-rate argument, so a line can be fitted to its slot without
    resampling artefacts."""

    name = "kokoro"
    default_voice = "af_heart"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or self._find_cache()

    @staticmethod
    def _find_cache() -> Path:
        for c in KOKORO_CACHE_CANDIDATES:
            if (c / "onnx-community").exists():
                return c
        return KOKORO_CACHE_CANDIDATES[-1]

    def available(self) -> bool:
        return (shutil.which("node") is not None
                and (KOKORO_DIR / "tts.mjs").exists()
                and (KOKORO_DIR / "node_modules" / "kokoro-js").exists())

    def synth(self, items: Sequence[Utterance], out_dir: Path, voice: str) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "outDir": str(out_dir),
            "cacheDir": str(self.cache_dir),
            "segments": [{"id": u.id, "text": u.text, "voice": voice or self.default_voice,
                          "speed": round(u.speed, 3)} for u in items],
        }
        proc = subprocess.run(["node", str(KOKORO_DIR / "tts.mjs")],
                              input=json.dumps(job).encode("utf-8"),
                              capture_output=True)
        if proc.returncode != 0:
            raise ToolError("kokoro tts failed: "
                            + proc.stderr.decode("utf-8", "replace")[-2000:])
        made: dict[str, Path] = {}
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log(f"ignoring unparseable tts worker output: {line[:200]!r}", level="warn")
                continue
            if rec.get("ok") and rec.get("id") and rec.get("file"):
                made[rec["id"]] = Path(rec["file"])
            else:
                log(f"tts failed for {rec.get('id')}: {rec.get('error')}", level="warn")
        return made


class SayTts(TtsBackend):
    """macOS `say`. Always present, no setup, lower quality."""

    name = "say"
    default_voice = "Samantha"
    _BASE_WPM = 175.0

    def available(self) -> bool:
        return shutil.which("say") is not None

    def synth(self, items: Sequence[Utterance], out_dir: Path, voice: str) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        made: dict[str, Path] = {}
        for u in items:
            aiff = out_dir / f"{u.id}.aiff"
            wav = out_dir / f"{u.id}.wav"
            rate = int(round(self._BASE_WPM * u.speed))
            try:
                run(["say", "-v", voice or self.default_voice, "-r", str(rate),
                     "-o", str(aiff), u.text])
                run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(aiff),
                     "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
                made[u.id] = wav
            except Exception as exc:
                log(f"tts failed for {u.id}: {exc}", level="warn")
            finally:
                aiff.unlink(missing_ok=True)
        return made


_BACKENDS = {"kokoro": KokoroTts, "say": SayTts}


def build(cfg: TextConfig) -> TtsBackend | None:
    if cfg.tts_backend == "none":
        return None
    names = list(_BACKENDS) if cfg.tts_backend == "auto" else [cfg.tts_backend]
    for name in names:
        cls = _BACKENDS.get(name)
        if cls is None:
            raise ToolError(f"unknown TTS backend: {name}")
        inst = cls()
        if inst.available():
            log(f"TTS backend: {inst.name}")
            return inst
    if cfg.tts_backend != "auto":
        raise ToolError(f"TTS backend {cfg.tts_backend!r} is not usable here")
    log("no TTS backend available; no dubbed track will be produced", level="warn")
    return None
