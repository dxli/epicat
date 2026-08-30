"""Command line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .pipeline import STAGES, Pipeline
from .util import ToolError, human_time, set_verbose

DESCRIPTION = """\
Join a set of episode clips into one file: order them by the episode number on
their title cards, drop the repeated title cards, erase the numbering from the
one that is kept, paint out burnt-in captions, and mux dual-language audio and
subtitles.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epicat", description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", type=str, help="input clips")
    p.add_argument("-o", "--output", default=None, help="output file (.mp4 or .mkv)")
    p.add_argument("-w", "--workdir", default=None,
                   help="scratch directory for intermediates (default: <output>.work)")
    p.add_argument("-c", "--config", default=None, help="TOML config file")
    p.add_argument("-v", "--verbose", action="store_true")

    p.add_argument("--inspect", action="store_true",
                   help="analyse the clips and print what would happen, then stop")
    p.add_argument("--force-from", choices=STAGES, default=None,
                   help="redo this stage and everything after it")

    g = p.add_argument_group("languages")
    g.add_argument("--source-lang", default=None, help="language of the clips (default: zh)")
    g.add_argument("--target-lang", default=None, help="language to add (default: en)")
    g.add_argument("--default-lang", default=None,
                   help="which audio/subtitle track is flagged default (default: en)")

    g = p.add_argument_group("titles")
    g.add_argument("--no-titles", action="store_true",
                   help="do not look for title cards at all")
    g.add_argument("--keep-all-titles", action="store_true",
                   help="keep every title card instead of only the first")

    g = p.add_argument_group("burnt-in captions")
    g.add_argument("--no-caption-removal", action="store_true",
                   help="leave burnt-in captions in the picture")
    g.add_argument("--band", default=None, metavar="TOP:BOTTOM",
                   help="caption band as fractions of frame height, e.g. 0.82:0.91")
    g.add_argument("--erase-region", action="append", default=[], metavar="X,Y,W,H",
                   help="also paint out this rectangle on every frame (fractions of "
                        "the frame, e.g. 0.79,0.90,0.21,0.08); repeatable")

    g = p.add_argument_group("text and audio")
    g.add_argument("--sub-source", choices=("ocr", "asr", "both"), default=None,
                   help="where subtitles come from (default: ocr, i.e. the burnt-in text)")
    g.add_argument("--asr-model", default=None, help="path to a whisper ggml model")
    g.add_argument("--translate-backend", default=None,
                   choices=("auto", "ollama", "passthrough", "none"))
    g.add_argument("--translate-model", default=None)
    g.add_argument("--tts-backend", default=None, choices=("auto", "kokoro", "say", "none"))
    g.add_argument("--tts-voice", default=None)
    g.add_argument("--no-dub", action="store_true",
                   help="do not synthesise a dubbed audio track")

    g = p.add_argument_group("encoding")
    g.add_argument("--crf", type=int, default=None)
    g.add_argument("--preset", default=None)
    g.add_argument("--vcodec", default=None)
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. --set band.min_px=60")

    return p


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.inputs:
        cfg.inputs = [str(Path(i)) for i in args.inputs]
    if args.output:
        cfg.output = args.output
    if not cfg.inputs:
        raise ToolError("no input clips given")

    if args.workdir:
        cfg.workdir = args.workdir
    elif cfg.workdir == Config().workdir:
        cfg.workdir = str(Path(cfg.output).with_suffix("").as_posix() + ".work")

    simple = {
        "source_lang": ("text.source_lang", args.source_lang),
        "target_lang": ("text.target_lang", args.target_lang),
        "default_lang": ("default_lang", args.default_lang),
        "sub_source": ("text.source", args.sub_source),
        "asr_model": ("text.asr_model", args.asr_model),
        "translate_backend": ("text.translate_backend", args.translate_backend),
        "translate_model": ("text.translate_model", args.translate_model),
        "tts_backend": ("text.tts_backend", args.tts_backend),
        "tts_voice": ("text.tts_voice", args.tts_voice),
        "crf": ("video.crf", args.crf),
        "preset": ("video.preset", args.preset),
        "vcodec": ("video.codec", args.vcodec),
    }
    overrides = {key: value for _, (key, value) in simple.items() if value is not None}

    if args.no_titles:
        overrides["title.enabled"] = False
    if args.keep_all_titles:
        overrides["title.keep_first"] = False
    if args.no_caption_removal:
        overrides["band.enabled"] = False
    if args.no_dub:
        overrides["text.tts_backend"] = "none"
    if args.band:
        top, _, bottom = args.band.partition(":")
        overrides["band.top"] = float(top)
        overrides["band.bottom"] = float(bottom)
    if args.erase_region:
        regions = []
        for spec in args.erase_region:
            parts = [p.strip() for p in spec.replace(":", ",").split(",")]
            if len(parts) != 4:
                raise ToolError(f"--erase-region expects X,Y,W,H, got {spec!r}")
            regions.append([float(p) for p in parts])
        overrides["band.extra_regions"] = regions

    for item in args.set:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ToolError(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = value.strip()

    cfg.apply_overrides(overrides)
    return cfg


def print_plan(pipe: Pipeline) -> None:
    plans = pipe.analyse()
    print()
    print(f"{'#':>2}  {'episode':>7}  {'title card':>12}  {'action':<22}  clip")
    print("-" * 92)
    for n, p in enumerate(plans):
        fps = p.media.fps_float
        card = f"{p.card.n_frames}f/{p.card.n_frames / fps:.2f}s" if p.card.present else "-"
        if p.cut_frames:
            action = "drop title card"
        elif p.erase_number:
            action = "keep, erase numbering"
        else:
            action = "keep as is"
        ep = p.episode if p.episode is not None else "?"
        print(f"{n + 1:>2}  {ep:>7}  {card:>12}  {action:<22}  {Path(p.media.path).name}")

    total = sum((p.media.nb_frames - p.cut_frames) / p.media.fps_float for p in plans)
    print("-" * 92)
    print(f"combined length ≈ {human_time(total)}")
    if plans[0].scan is not None:
        band = plans[0].scan.band
        capt = sum(int(p.scan.has_text.sum()) for p in plans if p.scan)
        frames = sum(p.scan.n_frames for p in plans if p.scan)
        runs = sum(len(p.scan.runs) for p in plans if p.scan)
        print(f"caption band  y={band.y0}..{band.y1} "
              f"({band.y0 / plans[0].media.height:.3f}..{band.y1 / plans[0].media.height:.3f})")
        print(f"captioned     {capt}/{frames} frames in {runs} stretches "
              f"({100 * capt / max(frames, 1):.0f}% of the running time)")
    else:
        print("caption band  not detected (removal disabled)")
    for w in pipe.warnings:
        print(f"warning: {w}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_verbose(args.verbose)
    try:
        cfg = make_config(args)
        pipe = Pipeline(cfg, force_from=args.force_from)
        if args.inspect:
            print_plan(pipe)
            return 0
        result = pipe.run()
    except ToolError as exc:
        print(f"epicat: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nepicat: interrupted", file=sys.stderr)
        return 130

    print()
    print(f"wrote {result.output}  ({human_time(result.duration)})")
    for c in result.clips:
        print(f"  ep {c.episode!s:>3}  {human_time(c.offset)}  {Path(c.path).name}"
              f"{'  [title card kept]' if c.kept_title else ''}")
    print(f"  subtitles: {result.source_srt.name}, {result.target_srt.name}")
    for w in result.warnings:
        print(f"  warning: {w}")
    return 0
