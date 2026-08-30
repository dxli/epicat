# epicat

Join a set of episode clips into one file.

`epicat` was built for a specific shape of source material: a series posted as
separate clips, each opening with a title card that carries an episode number
("第一集", "第二集", …), each with the dialogue burnt into the picture as
captions, and no separate audio or subtitle tracks for a second language.

Given those clips it will:

1. **Order them by episode number**, read off the title cards with OCR — the
   files themselves can be in any order, with any names.
2. **Keep one title card.** The first episode's card is kept, the rest are cut
   from the video and the audio together.
3. **Erase the numbering** from the card that is kept, so the result opens on a
   clean series title.
4. **Paint out the burnt-in captions** across every frame.
5. **Recover the caption text** while erasing it, and translate it.
6. **Mux two audio and two subtitle tracks** — the original language and a
   synthesised dub — with the language of your choice flagged as the default.

Everything runs locally. Nothing is uploaded.

## Quick start

```bash
python3 epicat.py clips/*.mp4 -o series.mp4
```

Look before you leap:

```bash
python3 epicat.py clips/*.mp4 -o series.mp4 --inspect
```

`--inspect` prints the episode order, the title-card boundaries it found, the
caption band it detected, and how much of the running time is captioned —
without touching a frame.

## What it produces

```
series.mp4        video + [en audio, zh audio] + [en subs, zh subs], en default
series.en.srt     the translated subtitles
series.zh.srt     the subtitles recovered from the picture
series.en.vtt     the same, for web players
series.zh.vtt
series.work/      intermediates, kept so a run can be resumed
```

## Requirements

| Need | Used for | Notes |
| --- | --- | --- |
| `ffmpeg` / `ffprobe` | everything | required on every platform |
| Python 3.11+ with `numpy` | frame processing | no other Python dependencies |
| `swiftc` (macOS) or `tesseract` (Linux/Windows) | OCR | macOS uses Apple Vision by default; the helper is built on first use |
| `ollama` | translation | optional — `--translate-backend passthrough` skips it |
| `node` | Kokoro speech synthesis | optional — falls back to macOS `say` where available |
| `whisper-cli` | speech recognition | optional — only for `--sub-source asr` |

`ffmpeg`, an OCR backend, and Python/`numpy` are what the core pipeline needs.
Everything else is optional and only used for the features it names.

### Bootstrap

```bash
python3 epicat.py --bootstrap
```

installs what's missing with the platform's own package manager — Homebrew on
macOS, whichever of `apt`, `dnf`/`yum`, `pacman`, `zypper`, or `apk` is present
on Linux, `winget` (or Chocolatey) on Windows — asking for your password or a
UAC prompt only when a step actually needs one. Nothing installs without your
say-so: each step is confirmed individually unless you pass `--yes`.

```bash
python3 epicat.py --bootstrap --check       # report what's missing, install nothing
python3 epicat.py --bootstrap --yes         # install the required pieces, no prompts
python3 epicat.py --bootstrap --optional    # also install translation, dubbing, ASR
python3 epicat.py --bootstrap --only ffmpeg,node   # limit to specific components
```

On macOS, if Homebrew itself is missing, bootstrap offers to install it first.
On Linux, Ollama has no reliable cross-distro package, so it's installed from
its own official script instead of the system package manager. A few pieces —
Xcode Command Line Tools on macOS, `whisper-cli` on Linux/Windows — have no
scriptable installer; bootstrap prints what to run or where to get them instead
of guessing.

`--bootstrap` works standalone: it never imports `numpy` or anything else
epicat depends on, so it runs even on a machine with nothing set up yet.

### Setting up the optional pieces by hand

```bash
# translation
ollama pull translategemma:27b

# better English speech than `say`
cd tools/kokoro && npm install

# speech recognition, only if the captions are not burnt in
mkdir -p ~/.cache/epicat/models
curl -L -o ~/.cache/epicat/models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

## How it works

### Finding the title card

A title card is a run of frames at the head of a clip that all show the same
glyphs. `epicat` takes a reference frame from the first couple of seconds and
then measures, for each following frame, how far the reference's glyph pixels
still stand above their immediate surroundings. That contrast decays smoothly
through a fade *and* through a cross-fade into the first shot, and collapses once
only artwork is left — so the same rule finds the boundary whether the card cuts
away hard or dissolves into the episode.

The card is then OCR'd. `第X集`, `第X话`, `EP07`, `Episode 12` and `Part 3` are
all recognised, and Chinese numerals are parsed up to 9999 (`第二十三集` → 23).
Clips whose number cannot be read keep their input order, at the end, with a
warning.

### Erasing the numbering

OCR gives the exact box of the numbering line. That box is filled with the
median colour of a ring drawn around it and feathered at the edges — which on a
title card, where the numbering sits on a flat plate, is exact. If the number
shares a line with other words, only its share of the line is erased.

### Erasing the captions

The caption band is found from the distribution of glyph-like pixels: captions
are solid, neutral white, and artwork almost never is, so a row histogram over
sampled frames makes the band obvious. Every frame's band is then decoded once —
cheaply, since it is a 40-pixel strip — to record which frames are captioned,
what the glyph mask is, and where the shots change.

Removing the glyphs is a two-step job:

- **Mask.** A strict brightness test finds the solid core of each stroke. That
  seed is grown by hysteresis over the anti-aliased edges, gated on a
  [top-hat transform][tophat] so the growth cannot flood across pale artwork —
  which happens constantly here, where white captions often sit on near-white
  paper.
- **Fill.** Where a clean frame exists in the same shot, its background is
  pasted in behind the glyphs. A donor whose surroundings no longer match the
  current frame is rejected, so a missed cut cannot produce a visibly wrong
  patch. Otherwise the hole is inpainted: seeded by a distance-weighted fill and
  relaxed towards the solution of the Laplace equation, solved with red-black
  over-relaxation on the mask's bounding box.

[tophat]: https://en.wikipedia.org/wiki/Top-hat_transform

### Recovering the text

The burnt-in captions *are* the script, so they are better source text than a
transcription of the audio — and their timing is frame-exact. `epicat` OCRs the
caption band a few times a second, applies a gamma curve first so pale
backgrounds darken while the white glyphs stay white, and folds the samples into
cues.

Captions here fade in over a second or more, so early frames read as fragments of
the final line. Two things handle that: samples are grouped by *containment* as
well as similarity, so a fragment joins the line it belongs to; and each sample
votes with a weight that rises steeply with how many glyph pixels were on screen,
so the fully drawn frame decides the text.

### Static overlays

A watermark or channel bug is a different problem from a caption: it is often
semi-transparent, so no single frame shows it clearly enough to threshold — but
it holds still while the artwork behind it moves. Averaging each pixel's
local-contrast response over sampled frames leaves the overlay standing and
averages the artwork away, which gives one mask for the whole clip.

This is off by default. Point it at a rectangle to switch it on:

```bash
python3 epicat.py clips/*.mp4 -o series.mp4 --erase-region 0.72,0.89,0.28,0.11
```

The rectangle is given as fractions of the frame — `x,y,width,height`. Expect the
area to be softened: what the overlay covered was never recorded.

### The dub

Cues are translated in numbered batches, so the model sees each line's
neighbours — which matters for Chinese, where the subject is often left out.

Each translated line is then synthesised, measured, and re-synthesised faster if
it would run into the next line. The result is laid onto a silent timeline at the
cue's own start time. The dubbed track carries dubbed speech and silence, nothing
else — the original-language audio is never mixed into it, since a player picking
a track by its language tag expects to hear only that language.

## Options

```
--inspect                 analyse and report, change nothing
--force-from STAGE        redo this stage and everything after it
                          (analyse render captions concat text translate
                           dub mux)

--bootstrap               install missing dependencies, then exit (see above)
--check                   with --bootstrap: report only, install nothing
--yes / -y                with --bootstrap: don't ask before each install
--optional                with --bootstrap: also install optional pieces
--only NAME,NAME,...      with --bootstrap: limit to these components

--source-lang / --target-lang / --default-lang
--no-titles               do not look for title cards
--keep-all-titles         keep every card instead of only the first
--no-caption-removal      leave the burnt-in captions alone
--band TOP:BOTTOM         set the caption band by hand, e.g. 0.82:0.91
--erase-region X,Y,W,H    also paint out this rectangle on every frame
                          (fractions of the frame); repeatable

--sub-source ocr|asr|both where subtitles come from (default: ocr)
--translate-backend auto|ollama|passthrough|none
--tts-backend auto|kokoro|say|none
--tts-voice NAME
--no-dub                  subtitles only, no second audio track

--crf / --preset / --vcodec
--set KEY=VALUE           override any config field, repeatable
```

Anything in the config can be set from the command line:

```bash
python3 epicat.py clips/*.mp4 -o out.mp4 \
  --set band.min_px=60 --set title.erase_margin=12
```

## Configuration file

```bash
python3 epicat.py -c series.toml
```

```toml
inputs = ["clips/a.mp4", "clips/b.mp4"]
output = "series.mp4"
default_lang = "en"

[title]
keep_first = true
erase_margin = 8

[band]
# Leave top/bottom unset to detect the band automatically.
top = 0.823
bottom = 0.907
min_px = 45
# Static overlays to paint out, as fractions of the frame: [x, y, w, h]
extra_regions = [[0.72, 0.89, 0.28, 0.11]]

[text]
source = "ocr"
translate_model = "translategemma:27b"
tts_backend = "auto"

[audio]
dub_max_speedup = 1.35

[video]
codec = "libx264"
crf = 17
```

See `epicat/config.py` for every field; each one is commented.

## Resuming

Intermediates live in the work directory, and a stage is skipped when its output
is already there. Every one of those files is written to a temporary name and
moved into place only once it is complete, so interrupting a run — Ctrl-C, a
crash, a closed laptop lid — never leaves a partial file for the next run to
mistake for a finished one; at worst you redo the one stage that was interrupted.

To redo part of a run — say the translation was poor, but the frame work was fine:

```bash
python3 epicat.py clips/*.mp4 -o series.mp4 --force-from translate
```

Re-reading the captions is a separate stage from rendering the frames, so
`--force-from captions` re-runs the OCR without re-encoding any video.

To hand-correct the subtitles, edit `series.work/source.zh.srt`, then re-run from
`translate`. To keep your own translation as well, edit `target.en.srt` and
re-run from `dub`.

## Notes and limits

- Clips must share a frame size. Differing frame rates are allowed but warned
  about, since timing can drift.
- Caption removal is inpainting, not reconstruction. Where a caption sat on
  detailed artwork with no clean frame in the shot to borrow from, the strip is
  softened. It is not possible to recover what was never recorded.
- The dubbed track is synthetic speech. It is a single voice, and it does not act.
- The Apple Vision OCR backend is macOS-only. Elsewhere, install `tesseract`
  along with `chi_sim` (or whichever language pack you need).

## License

MIT — see [LICENSE](LICENSE).
