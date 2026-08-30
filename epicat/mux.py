"""Final container assembly: one video, several audio and subtitle tracks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import AudioConfig
from .util import ToolError, log, run

# ISO 639-2/B codes; QuickTime and most players want these three-letter tags.
_ISO3 = {
    "en": "eng", "zh": "zho", "zh-CN": "zho", "zh-TW": "zho", "ja": "jpn",
    "ko": "kor", "fr": "fre", "de": "ger", "es": "spa", "ru": "rus", "it": "ita",
}

_TITLES = {"en": "English", "zh": "中文", "zh-CN": "简体中文", "zh-TW": "繁體中文",
           "ja": "日本語", "ko": "한국어"}


def iso3(code: str) -> str:
    return _ISO3.get(code, _ISO3.get(code.split("-")[0], code[:3]))


def track_title(code: str, suffix: str = "") -> str:
    base = _TITLES.get(code, _TITLES.get(code.split("-")[0], code))
    return f"{base} {suffix}".strip()


@dataclass
class Track:
    path: Path
    lang: str
    title: str = ""


def mux(video: Path, audio: Sequence[Track], subs: Sequence[Track], out: Path,
        *, default_lang: str, acfg: AudioConfig) -> None:
    """Write the finished file with language tags and default flags set.

    Tracks whose language is `default_lang` are listed first *and* flagged
    default: players are inconsistent about which they honour, so do both.
    """
    if not audio:
        raise ToolError("mux: at least one audio track is required")

    def order(tracks: Sequence[Track]) -> list[Track]:
        return sorted(tracks, key=lambda t: 0 if t.lang.split("-")[0] ==
                      default_lang.split("-")[0] else 1)

    audio = order(audio)
    subs = order(subs)

    container = out.suffix.lower()
    if container == ".mkv":
        sub_codec = "srt"
    elif container in (".mp4", ".m4v", ".mov"):
        sub_codec = "mov_text"
    else:
        raise ToolError(f"unsupported output container: {out.suffix}")

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video)]
    for t in list(audio) + list(subs):
        cmd += ["-i", str(t.path)]

    cmd += ["-map", "0:v:0"]
    for i in range(len(audio)):
        cmd += ["-map", f"{1 + i}:a:0"]
    for i in range(len(subs)):
        cmd += ["-map", f"{1 + len(audio) + i}:s:0"]

    cmd += ["-c:v", "copy", "-c:a", acfg.codec, "-b:a", acfg.bitrate,
            "-ar", str(acfg.sample_rate), "-ac", str(acfg.channels),
            "-c:s", sub_codec]

    for i, t in enumerate(audio):
        cmd += [f"-metadata:s:a:{i}", f"language={iso3(t.lang)}",
                f"-metadata:s:a:{i}", f"title={t.title or track_title(t.lang)}"]
    for i, t in enumerate(subs):
        cmd += [f"-metadata:s:s:{i}", f"language={iso3(t.lang)}",
                f"-metadata:s:s:{i}", f"title={t.title or track_title(t.lang)}"]

    for i, t in enumerate(audio):
        want = t.lang.split("-")[0] == default_lang.split("-")[0]
        cmd += [f"-disposition:a:{i}", "default" if want else "0"]
    for i, t in enumerate(subs):
        want = t.lang.split("-")[0] == default_lang.split("-")[0]
        cmd += [f"-disposition:s:{i}", "default" if want else "0"]

    if container in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd += [str(out)]

    log(f"muxing {len(audio)} audio and {len(subs)} subtitle tracks "
        f"(default: {default_lang}) → {out.name}")
    run(cmd)
