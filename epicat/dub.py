"""Building a dubbed audio track that lines up with a subtitle track."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AudioConfig
from .ffmpeg import audio_duration
from .subs import Cue
from .tts import TtsBackend, Utterance
from .util import atomic_output, log, run


def _read_wav(path: Path, rate: int) -> np.ndarray:
    """Decode any audio file to mono float32 at `rate`."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn",
         "-ar", str(rate), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _write_wav(path: Path, samples: np.ndarray, rate: int, channels: int) -> None:
    data = np.clip(samples, -1.0, 1.0)
    if channels == 2 and data.ndim == 1:
        data = np.stack([data, data], axis=1)
    pcm = (data * 32767.0).astype("<i2").tobytes()
    with atomic_output(path) as tmp:
        run(["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
             "-c:a", "pcm_s16le", str(tmp)], stdin=pcm)


def _slots(cues: Sequence[Cue], total: float) -> list[float]:
    """How long each line may run before it would collide with the next one."""
    out = []
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else total
        out.append(max(nxt - c.start, 0.3))
    return out


def synthesise(cues: Sequence[Cue], backend: TtsBackend, work: Path, voice: str,
               acfg: AudioConfig, total: float) -> dict[int, Path]:
    """Render every line, then re-render the ones that overrun their slot.

    A single corrected guess is not enough: speaking-rate control is not exactly
    proportional, so the fit is iterated. Each round measures what was actually
    produced and scales the rate by the residual, which converges in two or three
    passes for everything that is not already at the speed cap.
    """
    work.mkdir(parents=True, exist_ok=True)
    texts = {i: c.text.replace("\n", " ").strip() for i, c in enumerate(cues)}
    items = [Utterance(id=f"{i:05d}", text=t) for i, t in texts.items() if t]
    if not items:
        return {}

    log(f"synthesising {len(items)} lines with {backend.name}")
    made = backend.synth(items, work, voice)

    slots = _slots(cues, total)
    speeds: dict[str, float] = {u.id: 1.0 for u in items}

    for round_no in range(1, max(acfg.dub_fit_rounds, 0) + 1):
        retry: list[Utterance] = []
        for i, text in texts.items():
            key = f"{i:05d}"
            path = made.get(key)
            if path is None or not text:
                continue
            target = slots[i] * acfg.dub_fit_margin
            if target <= 0:
                continue
            dur = audio_duration(path)
            if dur <= target:
                continue
            wanted = min(speeds[key] * (dur / target), acfg.dub_max_speedup)
            if wanted <= speeds[key] * 1.01:
                continue        # already as fast as we are willing to go
            speeds[key] = wanted
            retry.append(Utterance(id=key, text=text, speed=wanted))
        if not retry:
            break
        log(f"fitting pass {round_no}: re-rendering {len(retry)} lines faster")
        made.update(backend.synth(retry, work, voice))

    return {int(k): v for k, v in made.items()}


def assemble(cues: Sequence[Cue], clips: dict[int, Path], out_wav: Path,
             total: float, acfg: AudioConfig) -> None:
    """Lay every rendered line onto a silent timeline at its cue start."""
    rate = acfg.sample_rate
    canvas = np.zeros(int(round(total * rate)) + rate, dtype=np.float32)
    slots = _slots(cues, total)
    overrun = 0

    for i, cue in enumerate(cues):
        path = clips.get(i)
        if path is None:
            continue
        audio = _read_wav(path, rate)
        if audio.size == 0:
            continue
        # A line that still overruns is trimmed with a short fade rather than
        # allowed to talk over the next one.
        limit = int(round(slots[i] * rate))
        if audio.size > limit:
            overrun += 1
            fade = min(int(0.05 * rate), limit // 4) or 1
            audio = audio[:limit].copy()
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        start = int(round(cue.start * rate))
        end = min(start + audio.size, canvas.size)
        if end > start:
            canvas[start:end] += audio[:end - start]

    peak = float(np.abs(canvas).max())
    if peak > 0:
        canvas *= min(1.0, 0.97 / peak)
    if overrun:
        log(f"{overrun} dubbed lines were trimmed to fit their slot", level="debug")
    _write_wav(out_wav, canvas[:int(round(total * rate))], rate, 1)


def mix_with_original(speech: Path, original: Path, out_path: Path,
                      acfg: AudioConfig) -> None:
    """Duck the original under the dub so music and effects survive.

    A sidechain compressor keyed on the speech track lowers the original only
    while someone is talking, which sounds far better than a flat gain cut.
    """
    duck = 10 ** (acfg.dub_duck_db / 20.0)
    gain = 10 ** (acfg.dub_gain_db / 20.0)
    ratio = max(1.0 / max(duck, 1e-3), 1.5)
    filt = (
        f"[1:a]aformat=channel_layouts=stereo,volume={gain:.4f}[speech];"
        f"[speech]asplit=2[sc][mix];"
        f"[0:a]aformat=channel_layouts=stereo[orig];"
        f"[orig][sc]sidechaincompress=threshold=0.03:ratio={min(ratio, 20):.2f}"
        f":attack=20:release=350:makeup=1[ducked];"
        f"[ducked][mix]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit=0.97[out]"
    )
    with atomic_output(out_path) as tmp:
        run(["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-i", str(original), "-i", str(speech),
             "-filter_complex", filt, "-map", "[out]",
             "-ar", str(acfg.sample_rate), "-ac", str(acfg.channels),
             "-c:a", "pcm_s16le", str(tmp)])
