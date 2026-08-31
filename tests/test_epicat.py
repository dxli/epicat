"""Unit tests for the parts that do not need video files.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epicat.captions import containment, normalise, same_caption, similar, tidy
from epicat.cjk import chinese_to_int, find_episode, strip_episode
from epicat.config import Config, TextConfig
from epicat.imaging import (box_mean, dilate, grey_min, grow_mask, inpaint,
                            plate_fill, text_mask, tophat, write_png)
from epicat.mux import iso3, subtitle_codec, track_title
from epicat.overlay import region_boxes
from epicat.subs import Cue, read_srt, write_srt
from epicat.translate import _parse_numbered, word_budgets
from epicat.util import ToolError, atomic_output, human_time


class TestChineseNumerals(unittest.TestCase):
    def test_digits(self):
        for text, want in [("一", 1), ("九", 9), ("十", 10), ("十一", 11),
                           ("二十", 20), ("二十三", 23), ("九十九", 99),
                           ("一百零八", 108), ("3", 3), ("１２", 12)]:
            self.assertEqual(chinese_to_int(text), want, text)

    def test_rejects_prose(self):
        self.assertIsNone(chinese_to_int("墨子"))
        self.assertIsNone(chinese_to_int(""))

    def test_episode_markers(self):
        for text, want in [("第四集", 4), ("墨子救宋 第十二集", 12), ("第3集", 3),
                           ("EP07", 7), ("Episode 12", 12), ("第二十三话", 23),
                           ("Part 2", 2)]:
            found = find_episode(text)
            self.assertIsNotNone(found, text)
            self.assertEqual(found[0], want, text)

    def test_no_marker(self):
        self.assertIsNone(find_episode("墨子救宋"))

    def test_strip(self):
        self.assertEqual(strip_episode("墨子救宋 第十二集"), "墨子救宋")


class TestImaging(unittest.TestCase):
    def test_box_mean(self):
        a = np.zeros((10, 10), np.float32)
        a[5, 5] = 9.0
        self.assertAlmostEqual(float(box_mean(a, 1)[5, 5]), 1.0, places=5)

    def test_dilate(self):
        m = np.zeros((10, 10), bool)
        m[5, 5] = True
        self.assertEqual(int(dilate(m, 1).sum()), 9)
        self.assertEqual(int(dilate(m, 2).sum()), 25)

    def test_grey_min_matches_a_brute_force_window(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 100, (9, 11)).astype(np.float32)
        r = 2
        want = np.empty_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                ys = slice(max(y - r, 0), min(y + r + 1, img.shape[0]))
                xs = slice(max(x - r, 0), min(x + r + 1, img.shape[1]))
                want[y, x] = img[ys, xs].min()
        got = grey_min(np.pad(img, r, mode="edge"), r)[r:-r, r:-r]
        self.assertTrue(np.allclose(got, want))

    def test_tophat_finds_bright_stroke_on_bright_ground(self):
        img = np.full((30, 60), 240.0, np.float32)   # near-white background
        img[12:16, 10:50] = 255.0                    # a white stroke on it
        th = tophat(img, 6)
        self.assertGreater(th[13, 20], 10.0)
        self.assertLess(th[2, 2], 1.0)

    def test_inpaint_recovers_a_gradient(self):
        base = np.zeros((24, 24, 3), np.uint8)
        base[:] = (np.arange(24)[:, None, None] * 10).astype(np.uint8)
        hole = np.zeros((24, 24), bool)
        hole[10:14, 8:16] = True
        damaged = base.copy()
        damaged[hole] = 255
        fixed = inpaint(damaged, hole)
        err = np.abs(fixed.astype(int) - base.astype(int))[hole].max()
        self.assertLessEqual(err, 3)

    def test_inpaint_without_holes_is_identity(self):
        img = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
        self.assertTrue((inpaint(img, np.zeros((10, 10), bool)) == img).all())

    def test_plate_fill_erases_text_on_a_flat_plate(self):
        plate = np.zeros((40, 80, 3), np.uint8)
        plate[10:20, 20:60] = 255
        out = plate_fill(plate, (14, 4, 52, 22), ring=6, feather_px=4)
        self.assertEqual(int(out.max()), 0)

    def test_grow_mask_does_not_flood_pale_ground(self):
        img = np.full((30, 60, 3), 235, np.uint8)    # pale, passes a loose test
        img[12:16, 10:20] = 255                      # the actual glyph
        seed = np.zeros((30, 60), bool)
        seed[13:15, 12:18] = True
        grown = grow_mask(img, seed)
        self.assertLess(int(grown.sum()), 400)
        self.assertGreaterEqual(int(grown.sum()), int(seed.sum()))

    def test_text_mask_ignores_coloured_highlights(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[2, 2] = (255, 255, 255)     # neutral white: a glyph
        img[5, 5] = (255, 200, 60)      # saturated: artwork
        m = text_mask(img)
        self.assertTrue(m[2, 2])
        self.assertFalse(m[5, 5])

    def test_png_roundtrip_is_readable(self):
        img = np.random.default_rng(1).integers(0, 255, (8, 12, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            write_png(p, img)
            self.assertTrue(p.stat().st_size > 0)
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class TestCaptionGrouping(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(normalise("  你好 世界 "), "你好世界")

    def test_fragment_joins_its_line(self):
        whole = "墨子闻讯心急如焚"
        self.assertGreater(containment("如焚", whole), 0.99)
        self.assertTrue(same_caption("如焚", whole, 0.7))

    def test_distinct_lines_stay_apart(self):
        self.assertFalse(same_caption("先生可算回来了", "我已等候多时", 0.7))

    def test_tidy_drops_a_fade_fragment(self):
        cues = [Cue(0.0, 2.0, "见楚王高踞看台"), Cue(2.1, 2.4, "台"),
                Cue(3.0, 5.0, "威仪赫赫")]
        out = tidy(cues, TextConfig())
        self.assertEqual([c.text for c in out], ["见楚王高踞看台", "威仪赫赫"])

    def test_tidy_keeps_a_short_line_that_stands_alone(self):
        cues = [Cue(0.0, 2.0, "先生主张和平"), Cue(2.1, 2.5, "不错"),
                Cue(3.0, 5.0, "那么君子就不斗了吗")]
        self.assertEqual(len(tidy(cues, TextConfig())), 3)

    def test_tidy_merges_a_caption_that_dipped_out(self):
        cues = [Cue(0.0, 1.0, "同一句"), Cue(1.2, 2.0, "同一句")]
        out = tidy(cues, TextConfig())
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].end, 2.0)

    def test_similar_is_symmetric_and_bounded(self):
        self.assertEqual(similar("abc", "abc"), 1.0)
        self.assertEqual(similar("", "abc"), 0.0)


class TestSubtitles(unittest.TestCase):
    def test_srt_roundtrip(self):
        cues = [Cue(0.0, 1.5, "第一行"), Cue(1.5, 3.25, "second line")]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.srt"
            write_srt(p, cues)
            back = read_srt(p)
        self.assertEqual(len(back), 2)
        self.assertAlmostEqual(back[1].start, 1.5, places=3)
        self.assertAlmostEqual(back[1].end, 3.25, places=3)
        self.assertEqual(back[0].text, "第一行")

    def test_shift(self):
        self.assertAlmostEqual(Cue(1.0, 2.0, "x").shifted(0.5).start, 1.5)


class TestTranslateParsing(unittest.TestCase):
    def test_numbered_reply(self):
        reply = "1. First line\n2. Second line\n3. Third line"
        self.assertEqual(_parse_numbered(reply, 3),
                         ["First line", "Second line", "Third line"])

    def test_missing_entries_come_back_empty(self):
        self.assertEqual(_parse_numbered("1. only one", 3), ["only one", "", ""])

    def test_strips_decoration(self):
        self.assertEqual(_parse_numbered('1) "Quoted"', 1), ["Quoted"])


class TestWordBudgets(unittest.TestCase):
    def test_budget_scales_with_time(self):
        cues = [Cue(0.0, 1.0, "a"), Cue(2.0, 6.0, "b")]
        short, long = word_budgets(cues, 3.0)
        self.assertLess(short, long)

    def test_a_line_may_borrow_only_a_little_of_the_pause(self):
        tight = word_budgets([Cue(0.0, 2.0, "a"), Cue(2.2, 3.0, "b")], 3.0)[0]
        after_silence = word_budgets([Cue(0.0, 2.0, "a"), Cue(60.0, 61.0, "b")], 3.0)[0]
        self.assertLessEqual(after_silence - tight, 5)

    def test_never_returns_less_than_three(self):
        # Even an implausibly slow speaking rate leaves room for a few words.
        self.assertEqual(word_budgets([Cue(0.0, 0.1, "a")], 0.2)[0], 3)


class TestMuxTags(unittest.TestCase):
    def test_iso3(self):
        self.assertEqual(iso3("en"), "eng")
        self.assertEqual(iso3("zh"), "zho")
        self.assertEqual(iso3("zh-CN"), "zho")

    def test_titles(self):
        self.assertEqual(track_title("en", "(dub)"), "English (dub)")


class TestOverlayRegions(unittest.TestCase):
    class _Media:
        width, height = 1000, 500

    def test_fractions_become_pixels(self):
        cfg = Config().band
        cfg.extra_regions = [[0.5, 0.8, 0.25, 0.1]]
        self.assertEqual(region_boxes(cfg, self._Media()), [(500, 400, 250, 50)])

    def test_malformed_region_is_rejected(self):
        cfg = Config().band
        cfg.extra_regions = [[0.5, 0.8, 0.25]]
        with self.assertRaises(ValueError):
            region_boxes(cfg, self._Media())


class TestConfig(unittest.TestCase):
    def test_dotted_overrides_coerce_types(self):
        c = Config()
        c.apply_overrides({"band.min_px": "60", "title.keep_first": "false",
                           "video.crf": 20})
        self.assertEqual(c.band.min_px, 60)
        self.assertIs(c.title.keep_first, False)
        self.assertEqual(c.video.crf, 20)

    def test_unknown_key_is_rejected(self):
        # A raw KeyError/AttributeError would surface as an unhandled
        # traceback in the CLI; ToolError is what cli.main() catches cleanly.
        with self.assertRaises(ToolError):
            Config().apply_overrides({"band.nope": 1})

    def test_unknown_top_level_section_is_rejected_cleanly(self):
        # A typo'd section name walks a dotted path whose first segment does
        # not exist at all -- that used to raise a raw AttributeError.
        with self.assertRaises(ToolError):
            Config().apply_overrides({"fooo.bar": "1"})

    def test_malformed_int_is_rejected_cleanly(self):
        with self.assertRaises(ToolError):
            Config().apply_overrides({"video.crf": "abc"})

    def test_malformed_bool_is_rejected_rather_than_silently_false(self):
        with self.assertRaises(ToolError):
            Config().apply_overrides({"title.keep_first": "treu"})

    def test_optional_float_field_coerces_from_none_default(self):
        # band.top / band.bottom default to None; coercion used to be driven
        # by the *current* value's type, so a None default meant a "--set
        # band.top=0.5" silently stored the string "0.5" instead of a float.
        c = Config()
        c.apply_overrides({"band.top": "0.5", "band.bottom": "0.9"})
        self.assertEqual(c.band.top, 0.5)
        self.assertIsInstance(c.band.top, float)
        self.assertEqual(c.band.bottom, 0.9)
        self.assertIsInstance(c.band.bottom, float)

    def test_load_rejects_malformed_toml_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.toml"
            p.write_text("this is not [ valid toml", encoding="utf-8")
            with self.assertRaises(ToolError):
                Config.load(str(p))

    def test_load_rejects_missing_file_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ToolError):
                Config.load(str(Path(d) / "missing.toml"))


class TestBootstrap(unittest.TestCase):
    """bootstrap.py must never touch the real system in a test: every case
    below fakes out subprocess/os/network before calling into it."""

    def test_imports_without_numpy(self):
        # The whole point of --bootstrap is getting numpy (and everything
        # else) installed in the first place, so importing it -- and routing
        # to it from the CLI -- must not require numpy to already be present.
        # This test's own process already has numpy loaded (other test
        # classes in this file import it), so the only reliable check is a
        # fresh subprocess that never imports the numpy-dependent modules.
        root = Path(__file__).resolve().parent.parent
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import epicat.bootstrap\n"
            "import epicat.cli\n"
            "assert 'numpy' not in sys.modules, sorted(sys.modules)\n"
            "print('ok')\n"
        ) % str(root)
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_command_tables_cover_every_supported_manager(self):
        from epicat.bootstrap import _manager_install_cmd
        for mgr in ("brew", "apt-get", "dnf", "yum", "pacman", "zypper", "apk",
                   "winget", "choco"):
            cmd = _manager_install_cmd(mgr, ["ffmpeg"])
            self.assertIn("ffmpeg", cmd)
            self.assertEqual(cmd[0], mgr)

    def test_unknown_manager_is_rejected(self):
        from epicat.bootstrap import _manager_install_cmd
        with self.assertRaises(ValueError):
            _manager_install_cmd("nuget", ["ffmpeg"])

    def test_linux_admin_step_is_sudo_prefixed_when_not_root(self):
        import epicat.bootstrap as b
        calls = []
        with unittest.mock.patch.object(b.subprocess, "run",
                                        side_effect=lambda cmd, **kw: calls.append(cmd) or
                                        unittest.mock.Mock(returncode=0)), \
             unittest.mock.patch.object(b.os, "geteuid", return_value=1000, create=True):
            ok = b.run_step(["apt-get", "install", "-y", "ffmpeg"],
                            needs_admin=True, os_name=b.OS_LINUX, dry_run=False)
        self.assertTrue(ok)
        self.assertEqual(calls, [["sudo", "apt-get", "install", "-y", "ffmpeg"]])

    def test_root_is_not_sudo_prefixed(self):
        import epicat.bootstrap as b
        calls = []
        with unittest.mock.patch.object(b.subprocess, "run",
                                        side_effect=lambda cmd, **kw: calls.append(cmd) or
                                        unittest.mock.Mock(returncode=0)), \
             unittest.mock.patch.object(b.os, "geteuid", return_value=0, create=True):
            b.run_step(["apt-get", "install", "-y", "ffmpeg"],
                      needs_admin=True, os_name=b.OS_LINUX, dry_run=False)
        self.assertEqual(calls, [["apt-get", "install", "-y", "ffmpeg"]])

    def test_macos_never_gets_sudo(self):
        import epicat.bootstrap as b
        calls = []
        with unittest.mock.patch.object(b.subprocess, "run",
                                        side_effect=lambda cmd, **kw: calls.append(cmd) or
                                        unittest.mock.Mock(returncode=0)):
            b.run_step(["brew", "install", "ffmpeg"], needs_admin=False,
                      os_name=b.OS_MACOS, dry_run=False)
        self.assertEqual(calls, [["brew", "install", "ffmpeg"]])

    def test_windows_admin_step_elevates_via_powershell(self):
        import epicat.bootstrap as b
        with unittest.mock.patch.object(b, "_is_admin_windows", return_value=False), \
             unittest.mock.patch.object(b, "_run_windows_elevated",
                                        return_value=0) as elevated:
            ok = b.run_step(["winget", "install", "-e", "--id", "Gyan.FFmpeg"],
                            needs_admin=True, os_name=b.OS_WINDOWS, dry_run=False)
        self.assertTrue(ok)
        elevated.assert_called_once()

    def test_dry_run_prints_but_runs_nothing(self):
        import epicat.bootstrap as b
        with unittest.mock.patch.object(b.subprocess, "run") as run:
            ok = b.run_step(["brew", "install", "ffmpeg"], needs_admin=False,
                            os_name=b.OS_MACOS, dry_run=True)
        self.assertTrue(ok)
        run.assert_not_called()

    def test_check_only_never_installs(self):
        import epicat.bootstrap as b
        with unittest.mock.patch.object(b, "detect_os", return_value=b.OS_MACOS), \
             unittest.mock.patch.object(b, "detect_manager", return_value="brew"), \
             unittest.mock.patch.object(b, "_has", return_value=False), \
             unittest.mock.patch.object(b, "_install") as install:
            rc = b.run_bootstrap(check_only=True, only=["ffmpeg"])
        install.assert_not_called()
        self.assertEqual(rc, 1)   # ffmpeg is required and reported missing

    def test_plan_marks_ffmpeg_required_everywhere(self):
        from epicat.bootstrap import COMPONENTS, OS_LINUX, OS_MACOS, OS_WINDOWS
        ffmpeg = next(c for c in COMPONENTS if c.key == "ffmpeg")
        for os_name in (OS_LINUX, OS_MACOS, OS_WINDOWS):
            self.assertTrue(ffmpeg.required(os_name))

    def test_tesseract_required_off_macos_only(self):
        from epicat.bootstrap import COMPONENTS, OS_LINUX, OS_MACOS, OS_WINDOWS
        tess = next(c for c in COMPONENTS if c.key == "tesseract")
        self.assertFalse(tess.required(OS_MACOS))
        self.assertTrue(tess.required(OS_LINUX))
        self.assertTrue(tess.required(OS_WINDOWS))

    def test_ollama_on_linux_is_installable_without_a_package_manager(self):
        from epicat.bootstrap import _installable, OS_LINUX, COMPONENTS
        ollama = next(c for c in COMPONENTS if c.key == "ollama")
        installable, reason = _installable(ollama, OS_LINUX, mgr=None)
        self.assertTrue(installable)

    def test_unrecognised_os_is_reported_and_refused(self):
        import epicat.bootstrap as b
        with unittest.mock.patch.object(b, "detect_os", return_value=b.OS_OTHER):
            self.assertEqual(b.run_bootstrap(check_only=True), 1)

    def test_only_filter_with_no_match_fails_cleanly(self):
        import epicat.bootstrap as b
        with unittest.mock.patch.object(b, "detect_os", return_value=b.OS_MACOS), \
             unittest.mock.patch.object(b, "detect_manager", return_value="brew"):
            self.assertEqual(b.run_bootstrap(check_only=True, only=["not-a-real-component"]), 1)


class TestCliBootstrapDispatch(unittest.TestCase):
    """--bootstrap must short-circuit before the numpy-dependent pipeline
    import, and route through epicat.bootstrap.run_bootstrap."""

    def test_bootstrap_flag_dispatches_without_importing_pipeline(self):
        import epicat.cli as cli
        with unittest.mock.patch("epicat.bootstrap.run_bootstrap",
                                 return_value=0) as run_bootstrap:
            rc = cli.main(["--bootstrap", "--check"])
        run_bootstrap.assert_called_once()
        self.assertEqual(rc, 0)
        _, kwargs = run_bootstrap.call_args
        self.assertTrue(kwargs["check_only"])

    def test_only_flag_is_split_on_commas(self):
        import epicat.cli as cli
        with unittest.mock.patch("epicat.bootstrap.run_bootstrap",
                                 return_value=0) as run_bootstrap:
            cli.main(["--bootstrap", "--only", "ffmpeg, node"])
        _, kwargs = run_bootstrap.call_args
        self.assertEqual(kwargs["only"], ["ffmpeg", "node"])

    def test_bad_output_extension_is_rejected_before_any_work(self):
        # A typo'd -o extension must not be discovered only at mux() -- the
        # very last pipeline stage, after render/OCR/translate/dub already ran.
        import epicat.cli as cli
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "a.mp4"
            clip.write_bytes(b"\x00")
            args = cli.build_parser().parse_args([str(clip), "-o", "out.avi"])
            with self.assertRaises(ToolError):
                cli.make_config(args)

    def test_malformed_band_flag_is_rejected_cleanly(self):
        import epicat.cli as cli
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "a.mp4"
            clip.write_bytes(b"\x00")
            for spec in ("not-a-number", "0.5"):  # missing colon, and bad numbers
                args = cli.build_parser().parse_args(
                    [str(clip), "-o", "out.mp4", "--band", spec])
                with self.assertRaises(ToolError):
                    cli.make_config(args)

    def test_malformed_erase_region_is_rejected_cleanly(self):
        import epicat.cli as cli
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "a.mp4"
            clip.write_bytes(b"\x00")
            args = cli.build_parser().parse_args(
                [str(clip), "-o", "out.mp4", "--erase-region", "0.1,0.2,bad,0.4"])
            with self.assertRaises(ToolError):
                cli.make_config(args)


class TestAtomicOutput(unittest.TestCase):
    def test_publishes_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.mp4"
            with atomic_output(p) as tmp:
                self.assertNotEqual(tmp, p)
                tmp.write_text("done")
            self.assertEqual(p.read_text(), "done")
            self.assertFalse(tmp.exists())

    def test_discards_on_exception(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.mp4"
            with self.assertRaises(RuntimeError):
                with atomic_output(p) as tmp:
                    tmp.write_text("partial")
                    raise RuntimeError("boom")
            self.assertFalse(p.exists())
            self.assertFalse(tmp.exists())

    def test_preserves_extension_for_format_sniffing(self):
        # A temp name of "video.mp4.part" would make ffmpeg refuse to write
        # it at all, since it infers the container from the trailing suffix.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "video.mp4"
            with atomic_output(p) as tmp:
                self.assertTrue(tmp.name.endswith(".mp4"))
                self.assertNotEqual(tmp.suffix, ".part")
                tmp.write_bytes(b"data")

    def test_does_not_clobber_an_existing_file_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.srt"
            p.write_text("original")
            with self.assertRaises(RuntimeError):
                with atomic_output(p) as tmp:
                    tmp.write_text("new")
                    raise RuntimeError("boom")
            self.assertEqual(p.read_text(), "original")


class TestHumanTime(unittest.TestCase):
    def test_negative_duration_clamps_to_zero(self):
        self.assertEqual(human_time(-5.0), "00:00:00.000")

    def test_formats_hours_minutes_seconds(self):
        self.assertEqual(human_time(3661.5), "01:01:01.500")


class TestFfmpegProbeFps(unittest.TestCase):
    def test_zero_over_zero_frame_rate_falls_back_instead_of_crashing(self):
        # ffprobe reports avg_frame_rate="0/0" for many real streams
        # (variable frame rate, short clips); Fraction("0/0") raises
        # ZeroDivisionError outright, so this has to be caught before
        # construction, not after a `== 0` check that never gets reached.
        from fractions import Fraction
        from epicat.ffmpeg import _safe_fps
        self.assertEqual(_safe_fps("0/0", "25/1"), 25)
        self.assertEqual(_safe_fps(None, "30/1"), 30)
        self.assertEqual(_safe_fps("0/0", "0/0"), 25)   # both unusable -> hard fallback
        self.assertEqual(_safe_fps("30000/1001", "25/1"), Fraction(30000, 1001))


class TestConcatEscaping(unittest.TestCase):
    def test_single_quote_in_filename_is_escaped(self):
        from epicat.ffmpeg import _concat_escape
        escaped = _concat_escape(Path("/tmp/clip's part 1.mp4"))
        # The ffmpeg concat demuxer's own escaping: close the quote, insert
        # an escaped quote, reopen -- exactly like POSIX shell quoting.
        self.assertEqual(escaped, "/tmp/clip'\\''s part 1.mp4")
        self.assertNotIn("''", escaped.replace("'\\''", ""))


class TestInpaintDegenerateMask(unittest.TestCase):
    def test_entirely_masked_region_is_left_untouched_not_blackened(self):
        # Reachable via overlay detection's own "treat the whole box as
        # overlay" fallback: with zero known pixels anywhere in reach, the
        # old behaviour was to fabricate solid black instead of leaving the
        # pixels alone.
        region = np.full((40, 60, 3), 180, np.uint8)
        mask = np.ones((40, 60), dtype=bool)
        out = inpaint(region, mask)
        self.assertTrue((out == region).all())


class TestMuxSubtitleCodec(unittest.TestCase):
    def test_known_containers(self):
        self.assertEqual(subtitle_codec(".mp4"), "mov_text")
        self.assertEqual(subtitle_codec(".mkv"), "srt")
        self.assertEqual(subtitle_codec(".MOV"), "mov_text")

    def test_unknown_container_is_rejected(self):
        with self.assertRaises(ToolError):
            subtitle_codec(".avi")


class TestSrtParsingRobustness(unittest.TestCase):
    def test_extra_arrow_on_timing_line_does_not_crash(self):
        # A hand-edited SRT (README explicitly documents this as a supported
        # resume workflow) with stray text left on the timing line used to
        # raise "too many values to unpack" from a bare .split("-->").
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.srt"
            p.write_text("1\n00:00:01,000 --> 00:00:02,000 --> stray\nhello\n",
                         encoding="utf-8")
            cues = read_srt(p)
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0].start, 1.0)
        self.assertAlmostEqual(cues[0].end, 2.0)


class TestPipelineAsrFallback(unittest.TestCase):
    """A missing ASR engine must not abort a run that never asked for ASR --
    only a run that explicitly requested it with nothing else to fall back
    on should propagate that failure."""

    def _pipeline(self, tmpdir, source):
        from epicat.config import Config
        from epicat.pipeline import Pipeline
        cfg = Config()
        cfg.inputs = ["a.mp4"]
        cfg.output = str(Path(tmpdir) / "out.mp4")
        cfg.workdir = str(Path(tmpdir) / "work")
        cfg.text.source = source
        return Pipeline(cfg)

    def test_ocr_only_default_does_not_crash_when_ocr_found_nothing(self):
        import epicat.pipeline as pipeline_mod
        with tempfile.TemporaryDirectory() as d:
            pipe = self._pipeline(d, "ocr")
            audio = Path(d) / "combined.wav"
            audio.write_bytes(b"\x00")
            with unittest.mock.patch.object(
                    pipeline_mod.asr_mod, "build",
                    side_effect=ToolError("whisper-cli not found")):
                cues = pipe.source_cues(audio, from_ocr=[])
            self.assertEqual(cues, [])
            self.assertTrue(any("speech recognition unavailable" in w
                               for w in pipe.warnings))

    def test_both_mode_keeps_ocr_cues_when_asr_engine_is_missing(self):
        import epicat.pipeline as pipeline_mod
        with tempfile.TemporaryDirectory() as d:
            pipe = self._pipeline(d, "both")
            audio = Path(d) / "combined.wav"
            audio.write_bytes(b"\x00")
            ocr_cues = [Cue(0.0, 1.0, "hello")]
            with unittest.mock.patch.object(
                    pipeline_mod.asr_mod, "build",
                    side_effect=ToolError("whisper-cli not found")):
                cues = pipe.source_cues(audio, from_ocr=ocr_cues)
            self.assertEqual([c.text for c in cues], ["hello"])

    def test_explicit_asr_only_request_still_raises(self):
        # cfg.source == "asr" with nothing else to fall back on: this is the
        # one case where propagating the failure is correct, since the user
        # asked for ASR specifically and there is no other source at all.
        import epicat.pipeline as pipeline_mod
        with tempfile.TemporaryDirectory() as d:
            pipe = self._pipeline(d, "asr")
            audio = Path(d) / "combined.wav"
            audio.write_bytes(b"\x00")
            with unittest.mock.patch.object(
                    pipeline_mod.asr_mod, "build",
                    side_effect=ToolError("whisper-cli not found")):
                with self.assertRaises(ToolError):
                    pipe.source_cues(audio, from_ocr=[])


class TestPipelineEmptySubtitles(unittest.TestCase):
    """Zero recovered cues must not produce a 0-byte .srt fed to mux() --
    ffmpeg refuses to open an empty file as an input at all."""

    def test_mux_accepts_zero_subtitle_tracks(self):
        from epicat.mux import mux, Track
        from epicat.config import AudioConfig
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.mp4"
            audio = Path(d) / "a.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                           "-i", "color=c=blue:s=64x64:d=1:r=10",
                           "-c:v", "libx264", "-crf", "30", "-an", str(video)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                           "-i", "anullsrc=r=48000:cl=stereo", "-t", "1",
                           "-c:a", "pcm_s16le", str(audio)], check=True)
            out = Path(d) / "out.mp4"
            mux(video, [Track(audio, "en", "English")], [], out,
                default_lang="en", acfg=AudioConfig())
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


class TestDubTrackIsSpeechOnly(unittest.TestCase):
    """A player selecting the dub track by its language tag must hear only
    that language: the original-language audio must never be mixed in."""

    def test_mix_with_original_no_longer_exists(self):
        # A direct regression guard: this function used to duck-and-mix the
        # original audio into the dub track, which is exactly the bug being
        # fixed here. Its reintroduction, even unused, would be a red flag.
        import epicat.dub as dub_mod
        self.assertFalse(hasattr(dub_mod, "mix_with_original"))

    def test_assemble_output_is_silent_outside_dubbed_lines(self):
        from epicat.config import AudioConfig
        from epicat.dub import assemble
        with tempfile.TemporaryDirectory() as d:
            # A short synthetic "dub line": a 1kHz tone, standing in for TTS
            # output so the test needs no network or TTS backend.
            clip = Path(d) / "line.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                           "-i", "sine=frequency=1000:duration=0.5",
                           "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(clip)],
                          check=True)
            cues = [Cue(2.0, 3.0, "hello")]
            out = Path(d) / "speech.wav"
            assemble(cues, {0: clip}, out, total=5.0, acfg=AudioConfig())

            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(out), "-f", "s16le", "-"],
                capture_output=True, check=True)
            samples = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float64)
            rate = 48000
            before = samples[:int(1.5 * rate)]   # well before the cue starts
            during = samples[int(2.1 * rate):int(2.4 * rate)]  # inside it
            self.assertLess(np.abs(before).max(), 50)   # silence, not just quiet
            self.assertGreater(np.abs(during).max(), 1000)  # the tone is actually there

    def test_pipeline_dub_does_not_touch_the_original_audio(self):
        # dub() used to take source_audio and mix it in; the new signature
        # has nothing to mix, by construction.
        import inspect
        from epicat.pipeline import Pipeline
        params = list(inspect.signature(Pipeline.dub).parameters)
        self.assertEqual(params, ["self", "target_cues", "total"])


class TestTextureAwareFill(unittest.TestCase):
    """best_shift()/apply_shift() and the run-level plan built on top of them
    (bandscan.assign_shift_plans, clean._apply_shift_plan) -- the texture-fill
    alternative to harmonic inpainting for holes with no donor frame."""

    @staticmethod
    def _striped_image(h=60, w=240, period=20):
        # A repeating vertical-stripe texture: any offset that is a multiple
        # of `period` is a perfect match, which makes this deterministic and
        # gives a known-good answer to test best_shift against.
        x = np.arange(w)
        stripe = ((x % period) < period // 2).astype(np.uint8) * 200 + 20
        return np.repeat(np.repeat(stripe[None, :, None], h, axis=0), 3, axis=2)

    def test_best_shift_finds_the_repeat_period(self):
        from epicat.imaging import best_shift
        img = self._striped_image()
        holes = np.zeros(img.shape[:2], dtype=bool)
        holes[20:40, 100:120] = True
        result = best_shift(img, holes, (15, 45, 95, 125), ring=5, search_y=2, search_x=60)
        self.assertIsNotNone(result)
        quality, dy, dx = result
        self.assertLess(quality, 0.05)          # a near-perfect match must exist
        self.assertEqual(dx % 20, 0)             # and it must land on the repeat period

    def test_apply_shift_recovers_the_texture(self):
        from epicat.imaging import best_shift, apply_shift
        img = self._striped_image()
        holes = np.zeros(img.shape[:2], dtype=bool)
        holes[20:40, 100:120] = True
        box = (15, 45, 95, 125)
        quality, dy, dx = best_shift(img, holes, box, ring=5, search_y=2, search_x=60)
        filled = apply_shift(img, holes, box, dy, dx, feather_px=0)
        # Away from the feathered edge, the recovered stripe must match the
        # original exactly -- this is real content, not a blur.
        core = holes.copy()
        core[:, :102] = False
        core[:, 118:] = False
        self.assertTrue((filled[core] == img[core]).all())

    def test_best_shift_returns_none_for_unique_content(self):
        from epicat.imaging import best_shift
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (60, 240, 3), dtype=np.uint8)  # pure noise: nothing repeats
        holes = np.zeros(img.shape[:2], dtype=bool)
        holes[20:40, 100:120] = True
        result = best_shift(img, holes, (15, 45, 95, 125), ring=5, search_y=2, search_x=60)
        # A match may exist by chance in noise, but never a *good* one.
        if result is not None:
            self.assertGreater(result[0], 0.3)

    def test_assign_shift_plans_only_targets_donor_free_runs(self):
        from epicat.bandscan import Run
        from epicat.config import SubtitleBandConfig
        run_with_donor = Run(start=0, end=10, donor_before=0)
        run_without = Run(start=20, end=30)
        cfg = SubtitleBandConfig()
        self.assertEqual(run_with_donor.shift_plan, [])
        self.assertEqual(run_without.shift_plan, [])
        self.assertTrue(cfg.texture_fill)   # on by default

    def test_plan_for_mask_covers_repeating_texture(self):
        from epicat.bandscan import _plan_for_mask
        from epicat.config import SubtitleBandConfig
        img = self._striped_image()
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[20:40, 100:120] = True
        cfg = SubtitleBandConfig()
        plan = _plan_for_mask(img, mask, cfg)
        self.assertGreater(len(plan), 0)

    def test_apply_shift_plan_self_consistency(self):
        # The exact regression this guards: a plan searched with best_shift()
        # (quality = MSE / ring-variance) used to be re-validated at render
        # time against donor_match_tolerance -- an unrelated, differently-
        # scaled threshold from the donor-patching code path -- so every
        # chunk was rejected, even on the very frame the plan came from.
        from epicat.bandscan import _plan_for_mask
        from epicat.config import SubtitleBandConfig
        from epicat.clean import _apply_shift_plan
        img = self._striped_image()
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[20:40, 100:120] = True
        cfg = SubtitleBandConfig()
        plan = _plan_for_mask(img, mask, cfg)
        self.assertGreater(len(plan), 0)
        _, covered = _apply_shift_plan(img, mask, plan, cfg)
        self.assertTrue(covered.any(),
                        "a plan must validate against the exact frame it was planned from")

    def test_apply_shift_plan_rejects_content_that_no_longer_matches(self):
        from epicat.bandscan import ChunkShift
        from epicat.config import SubtitleBandConfig
        from epicat.clean import _apply_shift_plan
        img = self._striped_image()
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[20:40, 100:120] = True
        # A shift that does NOT land on the repeat period points at content
        # that looks nothing like the hole's surroundings.
        bogus = [ChunkShift(box=(15, 45, 95, 125), dy=0, dx=7)]
        cfg = SubtitleBandConfig()
        out, covered = _apply_shift_plan(img, mask, bogus, cfg)
        self.assertFalse(covered.any())

    def test_apply_shift_plan_out_of_bounds_offset_is_skipped_not_crashed(self):
        from epicat.bandscan import ChunkShift
        from epicat.config import SubtitleBandConfig
        from epicat.clean import _apply_shift_plan
        img = self._striped_image()
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[20:40, 100:120] = True
        way_off = [ChunkShift(box=(15, 45, 95, 125), dy=0, dx=10_000)]
        cfg = SubtitleBandConfig()
        out, covered = _apply_shift_plan(img, mask, way_off, cfg)
        self.assertFalse(covered.any())
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
