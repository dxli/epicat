"""ffmpeg/ffprobe wrappers: probing, raw-frame streaming, raw-frame encoding."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .util import ToolError, log, need, run, run_text

BYTES_PER_PIXEL = {"rgb24": 3, "gray": 1}


@dataclass
class Media:
    path: str
    width: int
    height: int
    fps: Fraction
    duration: float
    nb_frames: int
    has_audio: bool
    sample_rate: int
    channels: int

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def probe(path: str | Path) -> Media:
    need("ffprobe")
    out = run_text([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ToolError(f"no video stream in {path}")
    a = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)

    fps = Fraction(v.get("avg_frame_rate") or "0/0")
    if fps == 0:
        fps = Fraction(v.get("r_frame_rate") or "25/1")
    duration = float(data["format"].get("duration") or v.get("duration") or 0.0)

    nb = v.get("nb_frames")
    nb_frames = int(nb) if nb and nb.isdigit() else int(round(duration * float(fps)))

    return Media(
        path=str(path),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
        has_audio=a is not None,
        sample_rate=int(a["sample_rate"]) if a else 0,
        channels=int(a["channels"]) if a else 0,
    )


def count_frames(path: str | Path) -> int:
    """Exact frame count (decodes the file; use only when the header count is untrusted)."""
    out = run_text([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(path),
    ]).strip()
    return int(out) if out.isdigit() else 0


def read_frames(
    path: str | Path,
    width: int,
    height: int,
    *,
    pix_fmt: str = "rgb24",
    vf: str | None = None,
    start: float | None = None,
    duration: float | None = None,
    threads: int = 0,
) -> Iterator[np.ndarray]:
    """Stream decoded frames as numpy arrays.

    `width`/`height` must be the dimensions *after* `vf` is applied.
    Yields (h, w, 3) uint8 for rgb24 and (h, w) uint8 for gray.
    """
    need("ffmpeg")
    bpp = BYTES_PER_PIXEL[pix_fmt]
    cmd: list[str] = ["ffmpeg", "-v", "error", "-nostdin"]
    if threads:
        cmd += ["-threads", str(threads)]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]

    fsz = width * height * bpp
    log("$ " + " ".join(cmd), level="debug")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=fsz * 2)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(fsz)
            if not buf or len(buf) < fsz:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            yield arr.reshape(height, width, bpp) if bpp > 1 else arr.reshape(height, width)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        err = proc.stderr.read() if proc.stderr else b""
        proc.wait()
        low = err.lower()
        expected = b"broken pipe" in low  # a caller that stops early closes the pipe
        if proc.returncode not in (0, None) and b"error" in low and not expected:
            log(f"decoder stderr: {err.decode('utf-8', 'replace')[-500:]}", level="warn")


def read_frames_at(path: str | Path, width: int, height: int, indices: Sequence[int],
                   *, pix_fmt: str = "rgb24", vf: str | None = None) -> dict[int, np.ndarray]:
    """Grab a specific, sorted set of frame indices in one sequential decode pass."""
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    out: dict[int, np.ndarray] = {}
    stop = wanted[-1]
    pos = 0
    it = iter(wanted)
    nxt = next(it)
    for frame in read_frames(path, width, height, pix_fmt=pix_fmt, vf=vf):
        if pos == nxt:
            out[pos] = frame.copy()
            try:
                nxt = next(it)
            except StopIteration:
                break
        pos += 1
        if pos > stop:
            break
    return out


class RawEncoder:
    """Pipe raw rgb24 frames into an ffmpeg encoder."""

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction,
                 *, vcodec: str = "libx264", crf: int = 17, preset: str = "medium",
                 pix_fmt: str = "yuv420p", extra: Sequence[str] = ()):
        need("ffmpeg")
        self.path = str(path)
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}",
            "-i", "-", "-an",
            "-c:v", vcodec,
        ]
        if vcodec.startswith("libx26"):
            cmd += ["-crf", str(crf), "-preset", preset]
        elif "videotoolbox" in vcodec:
            cmd += ["-q:v", str(max(1, min(100, 100 - crf * 2)))]
        cmd += ["-pix_fmt", pix_fmt, *extra, self.path]
        log("$ " + " ".join(cmd), level="debug")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read() if self.proc.stderr else b""
        rc = self.proc.wait()
        if rc != 0:
            raise ToolError(f"encoder failed ({rc}): {err.decode('utf-8', 'replace')[-2000:]}")

    def __enter__(self) -> "RawEncoder":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        else:
            try:
                self.proc.kill()
            except Exception:
                pass


def extract_audio(src: str | Path, dst: str | Path, *, start: float = 0.0,
                  duration: float | None = None, sample_rate: int = 48000,
                  channels: int = 2) -> None:
    """Decode a (sub)range of the source audio to WAV, or synthesise silence if there is none."""
    media = probe(src)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    if media.has_audio:
        if start:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", str(src)]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-map", "0:a:0"]
    else:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}"]
        cmd += ["-t", f"{duration if duration is not None else media.duration:.6f}"]
    cmd += ["-vn", "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)]
    run(cmd)


def concat_wavs(parts: Sequence[str | Path], dst: str | Path) -> None:
    if not parts:
        raise ToolError("concat_wavs: nothing to concatenate")
    inputs: list[str] = []
    for p in parts:
        inputs += ["-i", str(p)]
    filt = "".join(f"[{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[out]"
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", *inputs,
         "-filter_complex", filt, "-map", "[out]", "-c:a", "pcm_s16le", str(dst)])


def concat_videos(parts: Sequence[str | Path], dst: str | Path, workdir: Path) -> None:
    """Concatenate segments that share codec parameters, without re-encoding."""
    listing = workdir / "concat_list.txt"
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts), encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-an", str(dst)])


def audio_duration(path: str | Path) -> float:
    out = run_text(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(path)]).strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
