"""Translating subtitle cues. Ollama is the default backend; everything is
pluggable so a different engine can be dropped in."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Sequence

from .config import TextConfig
from .subs import Cue
from .util import ToolError, log

_LANG_NAMES = {
    "zh": "Chinese", "zh-CN": "Simplified Chinese", "zh-TW": "Traditional Chinese",
    "en": "English", "ja": "Japanese", "ko": "Korean", "fr": "French",
    "de": "German", "es": "Spanish", "ru": "Russian",
}


def lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, _LANG_NAMES.get(code.split("-")[0], code))


class Translator:
    name = "none"

    def available(self) -> bool:
        return False

    def translate(self, texts: Sequence[str], src: str, dst: str,
                  budgets: Sequence[int] | None = None) -> list[str]:
        """Translate `texts`. `budgets`, if given, is a soft word limit per line."""
        raise NotImplementedError


class OllamaTranslator(Translator):
    """Local Ollama server. Lines are sent in numbered batches so the model can
    use neighbouring dialogue as context, which matters for pronoun-light
    Chinese source text."""

    name = "ollama"

    def __init__(self, model: str, batch: int = 12, host: str = "http://localhost:11434",
                 timeout: float = 600.0):
        self.model = model
        self.batch = max(batch, 1)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as fh:
                tags = json.load(fh)
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return True
        log(f"ollama is running but has no model {self.model!r}; "
            f"available: {sorted(names)}", level="warn")
        return False

    def _generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                return json.load(fh).get("response", "")
        except urllib.error.URLError as exc:
            raise ToolError(f"ollama request failed: {exc}") from exc

    def translate(self, texts: Sequence[str], src: str, dst: str,
                  budgets: Sequence[int] | None = None) -> list[str]:
        out: list[str] = []
        total = (len(texts) + self.batch - 1) // self.batch
        for bi in range(total):
            lo, hi = bi * self.batch, (bi + 1) * self.batch
            chunk = list(texts[lo:hi])
            if not chunk:
                continue
            caps = list(budgets[lo:hi]) if budgets else None
            log(f"translating batch {bi + 1}/{total} ({len(chunk)} lines)", level="debug")
            out.extend(self._translate_batch(chunk, src, dst, caps))
        return out

    def _translate_batch(self, chunk: list[str], src: str, dst: str,
                         caps: list[int] | None) -> list[str]:
        if caps:
            numbered = "\n".join(f"{i + 1}. (max {caps[i]} words) {t}"
                                  for i, t in enumerate(chunk))
            budget_note = ("Each line shows the most words it may use — it has to be "
                           "spoken in the time the original takes. Stay within it; say "
                           "the same thing more briefly rather than dropping meaning.\n")
        else:
            numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
            budget_note = "Keep each line short enough to read as a subtitle.\n"
        prompt = (
            f"Translate these {lang_name(src)} subtitle lines into natural, spoken "
            f"{lang_name(dst)}.\n"
            "They are consecutive lines of one scene, so use the surrounding lines "
            "for context.\n"
            f"{budget_note}"
            "Keep each translation on one line.\n"
            "Reply with the same numbering and nothing else.\n\n"
            f"{numbered}\n"
        )
        reply = self._generate(prompt)
        parsed = _parse_numbered(reply, len(chunk))
        if parsed.count("") > len(chunk) // 2:
            # The batch came back unusable; fall back to one line at a time.
            log("batch translation unusable, retrying line by line", level="warn")
            parsed = []
            for line in chunk:
                single = self._generate(
                    f"Translate this {lang_name(src)} subtitle line into natural "
                    f"{lang_name(dst)}. Reply with the translation only.\n\n{line}\n")
                parsed.append(_clean(single))
        return [p or c for p, c in zip(parsed, chunk)]


class PassthroughTranslator(Translator):
    """Leave the text alone. Useful when the source is already in the target
    language, or when translations are supplied from a file."""

    name = "passthrough"

    def available(self) -> bool:
        return True

    def translate(self, texts: Sequence[str], src: str, dst: str,
                  budgets: Sequence[int] | None = None) -> list[str]:
        return list(texts)


_NUM = re.compile(r"^\s*(\d+)\s*[.)、:]\s*(.*)$")


def _clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-z]*\n?|```$", "", t, flags=re.MULTILINE).strip()
    t = re.sub(r"^(translation|english)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
    return t.strip(' "“”')


def _parse_numbered(reply: str, count: int) -> list[str]:
    got: dict[int, str] = {}
    current: int | None = None
    for raw in reply.splitlines():
        m = _NUM.match(raw)
        if m:
            current = int(m.group(1))
            got[current] = _clean(m.group(2))
        elif current is not None and raw.strip():
            got[current] = (got.get(current, "") + " " + _clean(raw)).strip()
    return [got.get(i + 1, "") for i in range(count)]


def build(cfg: TextConfig) -> Translator:
    backend = cfg.translate_backend
    if backend in ("none", "passthrough"):
        return PassthroughTranslator()
    if backend in ("auto", "ollama"):
        t = OllamaTranslator(cfg.translate_model, cfg.translate_batch)
        if t.available():
            return t
        if backend == "ollama":
            raise ToolError(
                f"ollama backend unavailable (model {cfg.translate_model!r}). "
                f"Start `ollama serve` and `ollama pull {cfg.translate_model}`, "
                "or pass --translate-backend passthrough.")
        log("no translation backend available; subtitles will not be translated",
            level="warn")
        return PassthroughTranslator()
    raise ToolError(f"unknown translation backend: {backend}")


def word_budgets(cues: Sequence[Cue], words_per_second: float,
                 max_borrow: float = 1.5) -> list[int]:
    """How many words each line may use to be speakable in its own slot.

    A line may borrow a little of the pause that follows it, but not a long
    silence — otherwise a line before a scene change gets an absurd allowance.
    """
    out: list[int] = []
    for i, c in enumerate(cues):
        gap = (cues[i + 1].start - c.end) if i + 1 < len(cues) else max_borrow
        span = c.duration + min(max(gap, 0.0), max_borrow)
        out.append(max(int(round(span * words_per_second)), 3))
    return out


def translate_cues(cues: Sequence[Cue], translator: Translator, src: str, dst: str,
                   *, words_per_second: float = 0.0) -> list[Cue]:
    if not cues:
        return []
    texts = [c.text.replace("\n", " ") for c in cues]
    budgets = word_budgets(cues, words_per_second) if words_per_second > 0 else None
    log(f"translating {len(texts)} cues {src}→{dst} via {translator.name}"
        + (" with a per-line word budget" if budgets else ""))
    done = translator.translate(texts, src, dst, budgets)
    return [Cue(c.start, c.end, t or c.text) for c, t in zip(cues, done)]
