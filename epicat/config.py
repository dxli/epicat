"""Configuration: defaults, TOML loading, CLI overrides."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class TitleConfig:
    """Detection of the leading title card and of the episode-number line."""
    enabled: bool = True
    scan_seconds: float = 12.0        # how far into a clip to look for the card
    dark_max_luma: float = 60.0       # a title card frame is mostly this dark
    bright_luma: int = 190            # what counts as a glyph pixel when matching
    min_bright_px: int = 200          # a reference title frame needs at least this much text
    end_ratio: float = 0.08           # card ends once glyph contrast falls to this share of its peak
    min_contrast: int = 40            # a real title card is this much brighter than its plate
    max_seconds: float = 20.0         # refuse to treat more than this as a title card
    erase_margin: int = 8             # px added around the OCR box before erasing
    erase_ring: int = 10              # px of surrounding plate sampled for the fill colour
    erase_feather: int = 4
    keep_first: bool = True           # keep episode 1's card, drop the rest
    langs: tuple[str, ...] = ("zh-Hans", "en-US")


@dataclass
class SubtitleBandConfig:
    """Detection and erasure of burnt-in captions."""
    enabled: bool = True
    # Band geometry as fractions of frame height; None means auto-detect.
    top: float | None = None
    bottom: float | None = None
    pad: int = 10                     # px added above/below the detected text envelope
    auto_samples: int = 24            # frames OCR'd when auto-detecting the band
    search_top: float = 0.55          # auto-detect only considers boxes below this
    min_luma: int = 238               # glyph pixels are near-white …
    max_sat: int = 16                 # … and neutral
    min_px: int = 45                  # glyph pixels needed to call a frame "captioned"
    grow_luma: int = 205              # hysteresis: stroke edges are at least this bright
    grow_sat: int = 45                # hysteresis: … and no more tinted than this
    grow_steps: int = 6
    grow_contrast: float = 12.0       # … and stand this far above the local background
    stroke: int = 6                   # px wider than a glyph stroke (top-hat window)
    min_run_frames: int = 4           # ignore shorter flickers
    merge_gap_frames: int = 3         # bridge sub-frame gaps within one caption
    dilate: int = 2                   # px the glyph mask grows after hysteresis
    feather: int = 3
    shot_break_delta: float = 6.0     # mean-colour jump in the band that ends a shot
    caption_change_iou: float = 0.55  # glyph overlap below this starts a new caption
    donor_search_frames: int = 240    # how far to look for a clean frame to copy from
    donor_match_tolerance: float = 9.0  # reject a donor whose surroundings differ by more
    # Extra rectangles to clear on every frame — a watermark, a channel bug —
    # as fractions of the frame size: [[x, y, w, h], …].
    extra_regions: list[list[float]] = field(default_factory=list)
    extra_min_luma: int = 165         # overlays are usually dimmer than captions
    extra_max_sat: int = 70
    extra_contrast: float = 8.0
    extra_persistence: float = 0.35   # share of the peak persistent response to call overlay


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    channels: int = 2
    codec: str = "aac"
    bitrate: str = "192k"
    # English dub mixing
    dub_duck_db: float = -11.0        # how far the original is pushed down under speech
    dub_gain_db: float = 1.0
    dub_max_speedup: float = 1.5      # cap on time-compressing a long line into its slot
    dub_min_speedup: float = 0.85
    dub_fit_rounds: int = 3           # measure-and-correct passes when fitting lines
    dub_fit_margin: float = 0.97      # aim slightly under the slot, not exactly at it


@dataclass
class TextConfig:
    source: str = "ocr"               # ocr | asr | both
    asr_backend: str = "whisper-cli"
    asr_model: str = ""               # path to a ggml model
    translate_backend: str = "ollama"
    translate_model: str = "translategemma:27b"
    translate_batch: int = 12
    translate_words_per_second: float = 3.0  # speaking rate used to budget line length; 0 disables
    tts_backend: str = "auto"         # auto | kokoro | say | none
    tts_voice: str = ""
    ocr_sample_hz: float = 4.0        # caption frames OCR'd per second of captioned video
    caption_similarity: float = 0.70  # OCR texts this alike are treated as one caption
    fragment_max_seconds: float = 0.6 # a cue this brief that repeats its neighbour is a fade artefact
    ocr_upscale: int = 4
    ocr_gamma: float = 3.0            # darkens pale backgrounds so white glyphs stand out
    source_lang: str = "zh"
    target_lang: str = "en"


@dataclass
class VideoConfig:
    codec: str = "libx264"
    crf: int = 17
    preset: str = "medium"
    pix_fmt: str = "yuv420p"


@dataclass
class Config:
    inputs: list[str] = field(default_factory=list)
    output: str = "output.mp4"
    workdir: str = ".epicat"
    default_lang: str = "en"          # which audio/subtitle track is flagged default
    title: TitleConfig = field(default_factory=TitleConfig)
    band: SubtitleBandConfig = field(default_factory=SubtitleBandConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
            _apply(cfg, data)
        return cfg

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted-key overrides, e.g. {"band.min_px": 60}."""
        for key, value in overrides.items():
            target: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                target = getattr(target, p)
            name = parts[-1]
            if not hasattr(target, name):
                raise KeyError(f"unknown config key: {key}")
            current = getattr(target, name)
            setattr(target, name, _coerce(value, current))


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(value, str):
        if isinstance(current, bool):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(current, int) and not isinstance(current, bool):
            return int(value)
        if isinstance(current, float):
            return float(value)
    return value


def _apply(obj: Any, data: dict) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, value)
