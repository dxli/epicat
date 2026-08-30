"""End-to-end orchestration.

    analyse → order → render → concat → text → translate → dub → mux

Every stage writes its result into the work directory, so a run can be resumed
or partially redone without repeating the expensive frame work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import asr as asr_mod
from . import captions as captions_mod
from . import dub as dub_mod
from . import translate as translate_mod
from . import tts as tts_mod
from .bandscan import Band, auto_band_multi, scan
from .clean import ClipPlan, RenderStats, render_clip
from .config import Config
from .ffmpeg import (Media, audio_duration, concat_videos, concat_wavs,
                     extract_audio, probe)
from .mux import Track, mux, track_title
from .ocr import Ocr
from .stages import STAGES
from .subs import Cue, read_srt, write_srt, write_vtt
from .titles import detect_card, read_card
from .util import ToolError, log, read_json, write_json


@dataclass
class ClipReport:
    path: str
    episode: int | None
    title_frames: int
    cut_frames: int
    kept_title: bool
    captions: int
    duration: float
    offset: float = 0.0
    stats: dict = field(default_factory=dict)


@dataclass
class Result:
    output: Path
    clips: list[ClipReport]
    source_srt: Path | None = None
    target_srt: Path | None = None
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: Config, *, force_from: str | None = None):
        self.cfg = cfg
        self.work = Path(cfg.workdir).expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.out = Path(cfg.output).expanduser().resolve()
        self.force_index = STAGES.index(force_from) if force_from else len(STAGES)
        self.warnings: list[str] = []
        self._ocr: Ocr | None = None

    # ------------------------------------------------------------------ utils

    def ocr(self) -> Ocr:
        if self._ocr is None:
            self._ocr = Ocr(langs=list(self.cfg.title.langs))
        return self._ocr

    def stale(self, stage: str) -> bool:
        return STAGES.index(stage) >= self.force_index

    def warn(self, msg: str) -> None:
        log(msg, level="warn")
        self.warnings.append(msg)

    # --------------------------------------------------------------- analysis

    def analyse(self) -> list[ClipPlan]:
        cfg = self.cfg
        inputs = [Path(p).expanduser() for p in cfg.inputs]
        missing = [p for p in inputs if not p.exists()]
        if missing:
            raise ToolError("input not found: " + ", ".join(str(m) for m in missing))
        if not inputs:
            raise ToolError("no input clips given")

        log(f"analysing {len(inputs)} clips")
        medias = [probe(p) for p in inputs]
        self._check_uniform_media(medias)

        band: Band | None = None
        if cfg.band.enabled:
            band = auto_band_multi(medias, cfg.band)
            if band is None:
                self.warn("could not locate a caption band; "
                          "burnt-in caption removal is off")

        # Scanning every band costs a decode pass per clip. It is only needed
        # if frames or captions are actually going to be (re)computed.
        want_scan = bool(band) and cfg.band.enabled and self._frames_needed(len(inputs))

        plans: list[ClipPlan] = []
        for i, (path, media) in enumerate(zip(inputs, medias)):
            card = read_card(media, detect_card(media, cfg.title), self.ocr(), cfg.title)
            if cfg.title.enabled and not card.present:
                self.warn(f"{path.name}: no title card found")
            elif card.episode is None:
                self.warn(f"{path.name}: title card has no readable episode number "
                          f"(read: {card.lines})")
            sc = scan(media, band, cfg.band) if want_scan else None
            plans.append(ClipPlan(media=media, card=card, scan=sc,
                                  episode=card.episode, index=i))

        return self.order(plans)

    def _frames_needed(self, count: int) -> bool:
        """True when some clip still has to be rendered or re-read."""
        if self.stale("render") or self.stale("captions"):
            return True
        seg = self.work / "segments"
        for n in range(count):
            if not (seg / f"{n:03d}.mp4").exists() or not (seg / f"{n:03d}.wav").exists():
                return True
            if self.cfg.text.source in ("ocr", "both") and \
                    not (seg / f"{n:03d}.cues.json").exists():
                return True
        return False

    def _check_uniform_media(self, medias: Sequence[Media]) -> None:
        first = medias[0]
        for m in medias[1:]:
            if (m.width, m.height) != (first.width, first.height):
                raise ToolError(
                    f"{Path(m.path).name} is {m.width}x{m.height} but "
                    f"{Path(first.path).name} is {first.width}x{first.height}; "
                    "clips must share a frame size")
            if abs(m.fps_float - first.fps_float) > 0.01:
                self.warn(f"{Path(m.path).name} runs at {m.fps_float:.3f} fps, "
                          f"not {first.fps_float:.3f}; timing may drift")

    def order(self, plans: list[ClipPlan]) -> list[ClipPlan]:
        """Sort by episode number; clips without one keep their input order last."""
        numbered = [p for p in plans if p.episode is not None]
        rest = [p for p in plans if p.episode is None]
        dupes = {p.episode for p in numbered
                 if sum(1 for q in numbered if q.episode == p.episode) > 1}
        if dupes:
            self.warn(f"duplicate episode numbers: {sorted(dupes)}")
        numbered.sort(key=lambda p: (p.episode, p.index))
        if rest:
            self.warn(f"{len(rest)} clip(s) without an episode number kept in input order")
        ordered = numbered + rest

        keep = self.cfg.title.keep_first
        for n, p in enumerate(ordered):
            p.erase_number = p.card.present and (n == 0 or not keep)
            p.cut_frames = p.card.n_frames if (keep and n > 0 and p.card.present) else 0
        log("play order: " + ", ".join(
            f"ep{p.episode if p.episode is not None else '?'}={Path(p.media.path).name}"
            for p in ordered))
        return ordered

    # ----------------------------------------------------------------- render

    def render(self, plans: Sequence[ClipPlan]
               ) -> tuple[list[Path], list[Path], list[ClipReport], list[Cue]]:
        seg_dir = self.work / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        videos: list[Path] = []
        audios: list[Path] = []
        reports: list[ClipReport] = []
        all_cues: list[Cue] = []
        offset = 0.0

        for n, plan in enumerate(plans):
            vid = seg_dir / f"{n:03d}.mp4"
            aud = seg_dir / f"{n:03d}.wav"
            cue_file = seg_dir / f"{n:03d}.cues.json"
            name = Path(plan.media.path).name

            if self.stale("render") or not vid.exists():
                log(f"[{n + 1}/{len(plans)}] rendering {name} "
                    f"(ep {plan.episode}, cutting {plan.cut_frames} title frames)")
                stats = render_clip(plan, vid, self.cfg)
                write_json(seg_dir / f"{n:03d}.stats.json", stats)
            else:
                log(f"[{n + 1}/{len(plans)}] reusing rendered {name}")
                stats = RenderStats(**read_json(seg_dir / f"{n:03d}.stats.json"))

            dur = stats.frames_written / plan.media.fps_float
            if self.stale("render") or not aud.exists():
                extract_audio(plan.media.path, aud, start=plan.cut_seconds,
                              duration=dur, sample_rate=self.cfg.audio.sample_rate,
                              channels=self.cfg.audio.channels)

            cues: list[Cue] = []
            if self.cfg.text.source in ("ocr", "both") and plan.scan is not None:
                if self.stale("captions") or not cue_file.exists():
                    raw = captions_mod.extract(plan.media, plan.scan, self.ocr(),
                                               self.cfg.text, self.cfg.band)
                    write_json(cue_file, [c.__dict__ for c in raw])
                else:
                    raw = [Cue(**d) for d in read_json(cue_file)]
                for c in raw:
                    start = c.start - plan.cut_seconds
                    end = c.end - plan.cut_seconds
                    if end <= 0.05:
                        continue
                    cues.append(Cue(max(start, 0.0) + offset, end + offset, c.text))

            videos.append(vid)
            audios.append(aud)
            reports.append(ClipReport(
                path=plan.media.path, episode=plan.episode,
                title_frames=plan.card.n_frames, cut_frames=plan.cut_frames,
                kept_title=plan.cut_frames == 0 and plan.card.present,
                captions=len(cues), duration=dur, offset=offset,
                stats=stats.__dict__))
            all_cues.extend(cues)
            offset += dur

        all_cues.sort(key=lambda c: c.start)
        return videos, audios, reports, captions_mod.tidy(all_cues, self.cfg.text)

    # ----------------------------------------------------------------- concat

    def concat(self, videos: Sequence[Path], audios: Sequence[Path]) -> tuple[Path, Path]:
        video = self.work / "combined.mp4"
        audio = self.work / "combined.source.wav"
        if self.stale("concat") or not video.exists():
            log(f"joining {len(videos)} cleaned segments")
            concat_videos(videos, video, self.work)
        if self.stale("concat") or not audio.exists():
            concat_wavs(audios, audio)
        return video, audio

    # ------------------------------------------------------------------- text

    def source_cues(self, audio: Path, from_ocr: Sequence[Cue]) -> list[Cue]:
        cfg = self.cfg.text
        srt = self.work / f"source.{cfg.source_lang}.srt"
        if not self.stale("text") and srt.exists():
            return read_srt(srt)

        cues = list(from_ocr)
        if cfg.source in ("asr", "both") or not cues:
            attempting_asr_as_fallback = cfg.source != "asr"
            if cfg.source == "ocr" and not cues:
                self.warn("no captions were recovered from the picture; "
                          "falling back to speech recognition")
            try:
                engine = asr_mod.build(cfg)
            except ToolError as exc:
                # `cfg.source == "asr"` is an explicit, sole request for ASR
                # with nothing to fall back to -- a missing engine there is a
                # real, actionable error. In every other case ASR is either
                # supplementing OCR ("both") or covering for OCR having found
                # nothing at all; a missing engine there should not discard
                # whatever OCR already produced (which may be nothing, but
                # even then the run should finish rather than abort).
                if not attempting_asr_as_fallback:
                    raise
                self.warn(f"speech recognition unavailable, continuing without it: {exc}")
                engine = None
            if engine is not None:
                heard = engine.transcribe(audio, self.work / "asr", cfg.source_lang)
                cues = self._prefer(cues, heard) if cues else heard
        if not cues:
            self.warn("no source subtitles could be produced")
        write_srt(srt, cues)
        return cues

    @staticmethod
    def _prefer(ocr_cues: list[Cue], asr_cues: list[Cue]) -> list[Cue]:
        """Keep OCR timing and text, and add spoken lines OCR never saw."""
        out = list(ocr_cues)
        for a in asr_cues:
            if not any(a.start < c.end and c.start < a.end for c in ocr_cues):
                out.append(a)
        out.sort(key=lambda c: c.start)
        return out

    def translate(self, cues: Sequence[Cue]) -> list[Cue]:
        cfg = self.cfg.text
        srt = self.work / f"target.{cfg.target_lang}.srt"
        if not self.stale("translate") and srt.exists():
            return read_srt(srt)
        if cfg.source_lang.split("-")[0] == cfg.target_lang.split("-")[0]:
            out = list(cues)
        else:
            out = translate_mod.translate_cues(
                cues, translate_mod.build(cfg), cfg.source_lang, cfg.target_lang,
                words_per_second=cfg.translate_words_per_second)
        write_srt(srt, out)
        return out

    # -------------------------------------------------------------------- dub

    def dub(self, target_cues: Sequence[Cue], source_audio: Path,
            total: float) -> Path | None:
        cfg = self.cfg.text
        mixed = self.work / f"audio.{cfg.target_lang}.wav"
        if not self.stale("dub") and mixed.exists():
            return mixed
        backend = tts_mod.build(cfg)
        if backend is None or not target_cues:
            return None
        clips = dub_mod.synthesise(target_cues, backend, self.work / "tts",
                                   cfg.tts_voice or backend.default_voice,
                                   self.cfg.audio, total)
        if not clips:
            self.warn("speech synthesis produced nothing; no dubbed track")
            return None
        speech = self.work / f"speech.{cfg.target_lang}.wav"
        dub_mod.assemble(target_cues, clips, speech, total, self.cfg.audio)
        dub_mod.mix_with_original(speech, source_audio, mixed, self.cfg.audio)
        return mixed

    # ------------------------------------------------------------------- main

    def run(self) -> Result:
        cfg = self.cfg
        plans = self.analyse()
        videos, audios, reports, ocr_cues = self.render(plans)
        video, source_audio = self.concat(videos, audios)

        total = audio_duration(source_audio)
        src_cues = self.source_cues(source_audio, ocr_cues)
        tgt_cues = self.translate(src_cues)

        subs_dir = self.out.parent
        subs_dir.mkdir(parents=True, exist_ok=True)
        stem = self.out.with_suffix("")
        src_srt = Path(f"{stem}.{cfg.text.source_lang}.srt")
        tgt_srt = Path(f"{stem}.{cfg.text.target_lang}.srt")
        write_srt(src_srt, src_cues)
        write_srt(tgt_srt, tgt_cues)
        write_vtt(Path(f"{stem}.{cfg.text.source_lang}.vtt"), src_cues)
        write_vtt(Path(f"{stem}.{cfg.text.target_lang}.vtt"), tgt_cues)

        dubbed = self.dub(tgt_cues, source_audio, total)

        audio_tracks = [Track(source_audio, cfg.text.source_lang,
                              track_title(cfg.text.source_lang, "(original)"))]
        if dubbed is not None:
            audio_tracks.append(Track(dubbed, cfg.text.target_lang,
                                      track_title(cfg.text.target_lang, "(dub)")))
        else:
            self.warn(f"no {cfg.text.target_lang} audio track was produced")

        # A 0-cue SRT is not a file ffmpeg can open at all (empty input), so an
        # empty track must be left out of the mux entirely rather than muxed
        # as a "subtitle track with no subtitles in it".
        sub_tracks: list[Track] = []
        if src_cues:
            sub_tracks.append(Track(src_srt, cfg.text.source_lang))
        if tgt_cues:
            sub_tracks.append(Track(tgt_srt, cfg.text.target_lang))

        default = cfg.default_lang.split("-")[0]
        for kind, tracks in (("audio", audio_tracks), ("subtitle", sub_tracks)):
            if not tracks:
                continue   # e.g. no subtitles were produced at all; nothing to flag
            if default not in {t.lang.split("-")[0] for t in tracks}:
                self.warn(f"no {cfg.default_lang} {kind} track to flag as default; "
                          f"the first {kind} track will be used instead")

        mux(video, audio_tracks, sub_tracks, self.out,
            default_lang=cfg.default_lang, acfg=cfg.audio)

        result = Result(output=self.out, clips=reports, source_srt=src_srt,
                        target_srt=tgt_srt, duration=total, warnings=self.warnings)
        write_json(self.work / "report.json", result)
        return result
