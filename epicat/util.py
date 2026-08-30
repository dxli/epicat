"""Small shared helpers: logging, subprocess, JSON state."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

_T0 = time.time()
VERBOSE = False


def set_verbose(v: bool) -> None:
    global VERBOSE
    VERBOSE = v


def log(msg: str, *, level: str = "info") -> None:
    if level == "debug" and not VERBOSE:
        return
    mark = {"info": "•", "warn": "!", "debug": "  ", "step": "▶"}.get(level, "•")
    print(f"[{time.time() - _T0:7.1f}s] {mark} {msg}", file=sys.stderr, flush=True)


class ToolError(RuntimeError):
    pass


def need(binary: str, hint: str = "") -> str:
    p = shutil.which(binary)
    if not p:
        raise ToolError(f"required program not found on PATH: {binary}" + (f" ({hint})" if hint else ""))
    return p


def run(cmd: Sequence[str], *, capture: bool = True, check: bool = True,
        stdin: Any = None, env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd), level="debug")
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture,
        text=False,
        input=stdin,
        env={**os.environ, **(env or {})} if env else None,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-4000:]
        raise ToolError(f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd[:6])} ...\n{err}")
    return proc


def run_text(cmd: Sequence[str], **kw) -> str:
    return run(cmd, **kw).stdout.decode("utf-8", "replace")


def _jsonable(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return {k: _jsonable(v) for k, v in asdict(o).items()}
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


@contextlib.contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path to write to; publish it onto `path` only if the
    `with` block finishes without raising.

    Every file a resumed run trusts by mere existence -- a rendered segment, a
    cue list, a combined track -- has to go through this: otherwise a process
    killed mid-write (Ctrl-C, a crash, an OOM kill) leaves a partial file that
    the *next* run mistakes for a finished one. `os.replace` is atomic on both
    POSIX and Windows, which is what makes the swap safe even in that case.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix has to survive the rename intact: ffmpeg (and other tools)
    # infer the output format from a file's extension, so a temp name of
    # "video.mp4.part" would make ffmpeg refuse to write it at all.
    tmp = path.with_name(path.stem + ".part" + path.suffix)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def human_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
