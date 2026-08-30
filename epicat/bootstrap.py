"""Bootstrap: detect and, on request, install the external tools epicat needs.

This module is standard-library only. It has to run *before* numpy (or
anything else epicat depends on) is necessarily installed, since getting that
sorted out is the point of `--bootstrap`.

Design:

  - A small registry of `Component`s, each with a presence check and, per
    package manager, the package name(s) that provide it.
  - The host OS and its package manager are auto-detected. Linux covers the
    common distro families (apt, dnf/yum, pacman, zypper, apk); Windows uses
    winget (falling back to Chocolatey if present); macOS uses Homebrew,
    which is itself installed on request if missing.
  - Package-manager commands that need elevated rights are run with `sudo` on
    Linux (skipped if already root) and, on Windows, re-launched elevated via
    PowerShell's `Start-Process -Verb RunAs`, which is what raises the UAC
    prompt. Homebrew on macOS is deliberately never run as root -- it manages
    its own privilege escalation for the few steps that need it, and the
    official installer script does the same.
  - Nothing is installed without the user's say-so: each step is confirmed
    interactively unless `--yes` was given, and `--check` never installs
    anything at all.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

OS_MACOS, OS_LINUX, OS_WINDOWS, OS_OTHER = "macos", "linux", "windows", "other"

LINUX_MANAGERS = ("apt-get", "dnf", "yum", "pacman", "zypper", "apk")
WINDOWS_MANAGERS = ("winget", "choco")

_KOKORO_DIR = Path(__file__).resolve().parent.parent / "tools" / "kokoro"


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


# --------------------------------------------------------------- platform


def detect_os() -> str:
    s = platform.system()
    if s == "Darwin":
        return OS_MACOS
    if s == "Linux":
        return OS_LINUX
    if s == "Windows":
        return OS_WINDOWS
    return OS_OTHER


def detect_manager(os_name: str) -> Optional[str]:
    """Which package manager bootstrap will use on this machine, if any."""
    if os_name == OS_MACOS:
        return "brew" if _has("brew") else None
    if os_name == OS_LINUX:
        for mgr in LINUX_MANAGERS:
            if _has(mgr):
                return mgr
        return None
    if os_name == OS_WINDOWS:
        for mgr in WINDOWS_MANAGERS:
            if _has(mgr):
                return mgr
        return None
    return None


def _is_admin_windows() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:
        return False


# --------------------------------------------------------------- running steps


def _fetch_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:  # nosec B310 - fixed, trusted URL
        return fh.read().decode("utf-8", "replace")


def _run_windows_elevated(cmd: Sequence[str]) -> int:
    """Re-launch `cmd` elevated via PowerShell. This is what raises the UAC
    prompt -- there is no non-interactive way to gain admin rights on Windows,
    so the user has to click through it."""
    quoted = ",".join("'" + a.replace("'", "''") + "'" for a in cmd[1:])
    ps = (f"Start-Process -FilePath '{cmd[0]}' "
          + (f"-ArgumentList {quoted} " if len(cmd) > 1 else "")
          + "-Verb RunAs -Wait")
    proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps])
    return proc.returncode


def run_step(cmd: Sequence[str], *, needs_admin: bool, os_name: str,
             dry_run: bool) -> bool:
    """Run one install command, elevating if the platform needs it.

    Output is not captured: these commands are interactive (they may prompt
    for a password), so they inherit the terminal directly.
    """
    cmd = list(cmd)
    _print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return True
    try:
        if needs_admin and os_name == OS_LINUX and os.geteuid() != 0:
            return subprocess.run(["sudo", *cmd]).returncode == 0
        if needs_admin and os_name == OS_WINDOWS and not _is_admin_windows():
            return _run_windows_elevated(cmd) == 0
        return subprocess.run(cmd).returncode == 0
    except FileNotFoundError as exc:
        _print(f"  ! {exc}")
        return False


def _manager_install_cmd(mgr: str, names: Sequence[str]) -> list[str]:
    if mgr == "brew":
        return ["brew", "install", *names]
    if mgr == "apt-get":
        return ["apt-get", "install", "-y", *names]
    if mgr in ("dnf", "yum"):
        return [mgr, "install", "-y", *names]
    if mgr == "pacman":
        return ["pacman", "-S", "--noconfirm", *names]
    if mgr == "zypper":
        return ["zypper", "--non-interactive", "install", *names]
    if mgr == "apk":
        return ["apk", "add", *names]
    if mgr == "winget":
        return ["winget", "install", "-e", "--id", names[0],
                "--accept-source-agreements", "--accept-package-agreements"]
    if mgr == "choco":
        return ["choco", "install", "-y", *names]
    raise ValueError(f"unsupported package manager: {mgr}")


# --------------------------------------------------------------- components


@dataclass
class Package:
    """Package names for one manager. `command`, if set, replaces the usual
    `<manager> install <names>` shape entirely (used for one-off installers)."""
    names: Sequence[str] = ()
    command: Optional[Sequence[str]] = None
    needs_admin: bool = False
    note: str = ""


@dataclass
class Component:
    key: str
    label: str
    check: Callable[[], bool]
    required_on: frozenset = frozenset()      # OS names where the pipeline needs this
    packages: dict = field(default_factory=dict)   # manager name -> Package
    manual: dict = field(default_factory=dict)     # OS name -> instructions, no package exists
    post: Optional[Callable[[bool], bool]] = None  # extra step after install; takes dry_run

    def required(self, os_name: str) -> bool:
        return os_name in self.required_on


def _safe_check(comp: Component) -> bool:
    try:
        return comp.check()
    except Exception:
        return False


def _check_tesseract_chi_sim() -> bool:
    if not _has("tesseract"):
        return False
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                             text=True, timeout=10)
        return "chi_sim" in out.stdout
    except Exception:
        return False


def _install_kokoro_deps(dry_run: bool) -> bool:
    if not (_KOKORO_DIR / "package.json").exists():
        return True
    if (_KOKORO_DIR / "node_modules" / "kokoro-js").exists():
        return True
    _print(f"  $ npm install (in {_KOKORO_DIR})")
    if dry_run:
        return True
    proc = subprocess.run(["npm", "install", "--no-audit", "--no-fund"],
                          cwd=str(_KOKORO_DIR))
    return proc.returncode == 0


COMPONENTS: list[Component] = [
    Component(
        key="ffmpeg", label="ffmpeg / ffprobe",
        check=lambda: _has("ffmpeg") and _has("ffprobe"),
        required_on=frozenset({OS_MACOS, OS_LINUX, OS_WINDOWS}),
        packages={
            "brew": Package(names=["ffmpeg"]),
            "apt-get": Package(names=["ffmpeg"], needs_admin=True),
            "dnf": Package(names=["ffmpeg"], needs_admin=True,
                           note="plain Fedora needs the RPM Fusion repo enabled first"),
            "yum": Package(names=["ffmpeg"], needs_admin=True,
                           note="plain RHEL/CentOS needs the RPM Fusion repo enabled first"),
            "pacman": Package(names=["ffmpeg"], needs_admin=True),
            "zypper": Package(names=["ffmpeg"], needs_admin=True),
            "apk": Package(names=["ffmpeg"], needs_admin=True),
            "winget": Package(names=["Gyan.FFmpeg"]),
            "choco": Package(names=["ffmpeg"], needs_admin=True),
        },
    ),
    Component(
        key="xcode-clt", label="Xcode Command Line Tools (Apple Vision OCR)",
        check=lambda: _has("swiftc"),
        required_on=frozenset(),   # macOS can fall back to tesseract instead
        manual={OS_MACOS: "run `xcode-select --install`; this opens a GUI installer -- "
                          "finish it, then re-run --bootstrap"},
    ),
    Component(
        key="tesseract", label="Tesseract OCR + Chinese (Simplified) data",
        check=_check_tesseract_chi_sim,
        required_on=frozenset({OS_LINUX, OS_WINDOWS}),
        packages={
            "brew": Package(names=["tesseract", "tesseract-lang"]),
            "apt-get": Package(names=["tesseract-ocr", "tesseract-ocr-chi-sim"], needs_admin=True),
            "dnf": Package(names=["tesseract", "tesseract-langpack-chi_sim"], needs_admin=True),
            "yum": Package(names=["tesseract", "tesseract-langpack-chi_sim"], needs_admin=True),
            "pacman": Package(names=["tesseract", "tesseract-data-chi_sim"], needs_admin=True),
            "zypper": Package(names=["tesseract-ocr", "tesseract-ocr-traineddata-chinese_simplified"],
                              needs_admin=True),
            "apk": Package(names=["tesseract-ocr", "tesseract-ocr-data-chi_sim"], needs_admin=True),
            "winget": Package(names=["UB-Mannheim.TesseractOCR"],
                              note="chi_sim data may need a manual download into the Tesseract "
                                   "tessdata folder -- see https://github.com/tesseract-ocr/tessdata"),
        },
    ),
    Component(
        key="ollama", label="Ollama (translation backend)",
        check=lambda: _has("ollama"),
        packages={
            "brew": Package(names=["ollama"]),
            "winget": Package(names=["Ollama.Ollama"]),
        },
        manual={OS_LINUX: "installed via the official script (curl | sh), not a package manager"},
    ),
    Component(
        key="node", label="Node.js (Kokoro text-to-speech)",
        check=lambda: _has("node") and _has("npm"),
        packages={
            "brew": Package(names=["node"]),
            "apt-get": Package(names=["nodejs", "npm"], needs_admin=True),
            "dnf": Package(names=["nodejs", "npm"], needs_admin=True),
            "yum": Package(names=["nodejs", "npm"], needs_admin=True),
            "pacman": Package(names=["nodejs", "npm"], needs_admin=True),
            "zypper": Package(names=["nodejs", "npm"], needs_admin=True),
            "apk": Package(names=["nodejs", "npm"], needs_admin=True),
            "winget": Package(names=["OpenJS.NodeJS.LTS"]),
            "choco": Package(names=["nodejs-lts"], needs_admin=True),
        },
        post=_install_kokoro_deps,
    ),
    Component(
        key="whisper-cli", label="whisper.cpp (speech-recognition fallback)",
        check=lambda: _has("whisper-cli"),
        packages={"brew": Package(names=["whisper-cpp"])},
        manual={
            OS_LINUX: "no standard package; build from source: "
                     "https://github.com/ggml-org/whisper.cpp#quick-start",
            OS_WINDOWS: "no standard package; build from source or grab a release: "
                       "https://github.com/ggml-org/whisper.cpp#quick-start",
        },
    ),
]


# --------------------------------------------------------- special installers


def ensure_homebrew(*, dry_run: bool, assume_yes: bool) -> bool:
    if _has("brew"):
        return True
    _print("Homebrew is required to install packages on macOS but was not found.")
    if not (assume_yes or _confirm("Install Homebrew now?")):
        return False
    url = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
    _print(f"  fetching {url}")
    if dry_run:
        _print("  $ /bin/bash -c \"$(curl -fsSL " + url + ")\"")
        return True
    try:
        script = _fetch_text(url)
    except (urllib.error.URLError, TimeoutError) as exc:
        _print(f"  ! could not download the Homebrew installer: {exc}")
        return False
    _print("  running the official Homebrew installer (it will explain each step, "
          "and may ask for your password)")
    proc = subprocess.run(["/bin/bash", "-c", script])
    return proc.returncode == 0


def install_xcode_clt(*, dry_run: bool) -> bool:
    _print("  $ xcode-select --install")
    if dry_run:
        return True
    subprocess.run(["xcode-select", "--install"])
    _print("  a software-update window should have opened -- finish the install, "
          "then re-run --bootstrap")
    return True


def install_ollama_linux(*, dry_run: bool, assume_yes: bool) -> bool:
    url = "https://ollama.com/install.sh"
    _print(f"  installing Ollama via the official script ({url})")
    if not (assume_yes or _confirm("Proceed?")):
        return False
    if dry_run:
        _print(f"  $ curl -fsSL {url} | sh")
        return True
    try:
        script = _fetch_text(url)
    except (urllib.error.URLError, TimeoutError) as exc:
        _print(f"  ! could not download the Ollama installer: {exc}")
        return False
    # The script elevates itself with sudo where it needs to, so it is run
    # as-is rather than pre-empted with our own sudo.
    proc = subprocess.run(["/bin/sh", "-c", script])
    return proc.returncode == 0


# ------------------------------------------------------------------- planning


@dataclass
class PlanItem:
    component: Component
    present: bool
    required: bool
    installable: bool
    reason: str = ""


def _installable(comp: Component, os_name: str, mgr: Optional[str]) -> tuple:
    if comp.key == "xcode-clt":
        return (os_name == OS_MACOS), ("" if os_name == OS_MACOS else "not applicable here")
    if comp.key == "ollama" and os_name == OS_LINUX:
        return True, ""
    if mgr and mgr in comp.packages:
        return True, ""
    if os_name in comp.manual:
        return False, comp.manual[os_name]
    return False, "no install method known for this platform / package manager"


def build_plan(os_name: str, mgr: Optional[str], only: Optional[Sequence[str]]) -> list:
    items = []
    for comp in COMPONENTS:
        if only and comp.key not in only:
            continue
        installable, reason = _installable(comp, os_name, mgr)
        items.append(PlanItem(
            component=comp, present=_safe_check(comp), required=comp.required(os_name),
            installable=installable, reason=reason))
    return items


def _install(comp: Component, os_name: str, mgr: Optional[str], *,
            dry_run: bool, assume_yes: bool) -> bool:
    if comp.key == "xcode-clt":
        return install_xcode_clt(dry_run=dry_run)
    if comp.key == "ollama" and os_name == OS_LINUX:
        return install_ollama_linux(dry_run=dry_run, assume_yes=assume_yes)
    if not mgr or mgr not in comp.packages:
        _print(f"  no install method for {comp.label} on this platform")
        return False
    pkg = comp.packages[mgr]
    if pkg.note:
        _print(f"  note: {pkg.note}")
    cmd = list(pkg.command) if pkg.command else _manager_install_cmd(mgr, pkg.names)
    return run_step(cmd, needs_admin=pkg.needs_admin, os_name=os_name, dry_run=dry_run)


# ---------------------------------------------------------------------- entry


def run_bootstrap(*, check_only: bool = False, assume_yes: bool = False,
                  only: Optional[Sequence[str]] = None,
                  include_optional: bool = False, dry_run: bool = False) -> int:
    os_name = detect_os()
    if os_name == OS_OTHER:
        _print(f"unrecognised OS ({platform.system()}); bootstrap only knows "
              "macOS, Linux, and Windows")
        return 1

    mgr = detect_manager(os_name)
    _print(f"platform: {os_name}    package manager: {mgr or '(none found)'}")
    _print()

    if os_name == OS_MACOS and mgr is None and not check_only:
        if ensure_homebrew(dry_run=dry_run, assume_yes=assume_yes):
            mgr = "brew"
        _print()

    plan = build_plan(os_name, mgr, only)
    if not plan:
        _print(f"nothing matched --only {','.join(only or [])!r}")
        return 1

    _print(f"{'component':<42} {'status':<10} action")
    _print("-" * 80)
    for item in plan:
        status = "present" if item.present else ("required" if item.required else "optional")
        action = "-" if item.present else ("install" if item.installable
                                           else f"manual: {item.reason}")
        _print(f"{item.component.label:<42} {status:<10} {action}")
    _print()

    if check_only:
        missing_required = [i for i in plan if i.required and not i.present]
        if missing_required:
            _print(f"{len(missing_required)} required component(s) missing.")
            return 1
        _print("everything required is present.")
        return 0

    failures: list[str] = []
    for item in plan:
        if item.present:
            continue
        if not item.required and not include_optional and only is None:
            _print(f"skipping optional: {item.component.label} "
                  "(pass --optional to include it, or name it with --only)")
            continue
        if not item.installable:
            _print(f"skipping {item.component.label}: {item.reason}")
            continue
        _print(f"--- {item.component.label} ---")
        if not assume_yes and not _confirm(f"Install {item.component.label} now?"):
            _print("  skipped")
            continue
        ok = _install(item.component, os_name, mgr, dry_run=dry_run, assume_yes=assume_yes)
        if ok and item.component.post:
            ok = item.component.post(dry_run)
        if not ok:
            failures.append(item.component.label)
        _print()

    if failures:
        _print("failed to install: " + ", ".join(failures))
        return 1
    _print("bootstrap complete. Run `--bootstrap --check` to verify.")
    return 0
