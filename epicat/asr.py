"""Speech recognition, used when captions are not burnt into the picture."""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import TextConfig
from .subs import Cue, read_srt
from .util import ToolError, log, run

MODEL_SEARCH = (
    Path.home() / ".cache" / "epicat" / "models",
    Path.home() / ".cache" / "whisper.cpp",
    Path("/opt/homebrew/share/whisper.cpp"),
    Path("/usr/local/share/whisper.cpp"),
)


def find_model(explicit: str = "") -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise ToolError(f"ASR model not found: {p}")
    for d in MODEL_SEARCH:
        if not d.exists():
            continue
        # Prefer the largest model present: it is usually the best one.
        found = sorted(d.glob("ggml-*.bin"), key=lambda p: p.stat().st_size, reverse=True)
        if found:
            return found[0]
    return None


class WhisperCli:
    """whisper.cpp's `whisper-cli`."""

    name = "whisper-cli"

    def __init__(self, model: Path):
        self.model = model

    @staticmethod
    def available() -> bool:
        return shutil.which("whisper-cli") is not None

    def transcribe(self, audio: Path, work: Path, lang: str) -> list[Cue]:
        work.mkdir(parents=True, exist_ok=True)
        wav = work / "asr16k.wav"
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(audio),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        stem = work / "asr"
        log(f"transcribing with {self.model.name} (language={lang})")
        run(["whisper-cli", "-m", str(self.model), "-f", str(wav),
             "-l", lang, "-osrt", "-of", str(stem), "-pp"], capture=True)
        srt = stem.with_suffix(".srt")
        if not srt.exists():
            raise ToolError("whisper-cli produced no subtitle output")
        return read_srt(srt)


def build(cfg: TextConfig):
    if cfg.asr_backend in ("none", ""):
        return None
    if cfg.asr_backend not in ("auto", "whisper-cli"):
        raise ToolError(f"unknown ASR backend: {cfg.asr_backend}")
    if not WhisperCli.available():
        raise ToolError("whisper-cli not found on PATH (brew install whisper-cpp)")
    model = find_model(cfg.asr_model)
    if model is None:
        raise ToolError(
            "no whisper model found. Download one, for example:\n"
            "  mkdir -p ~/.cache/epicat/models && curl -L -o "
            "~/.cache/epicat/models/ggml-large-v3-turbo.bin \\\n"
            "    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            "ggml-large-v3-turbo.bin\n"
            "or point --asr-model at an existing ggml-*.bin file.")
    return WhisperCli(model)
