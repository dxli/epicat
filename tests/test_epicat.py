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
from epicat.mux import iso3, track_title
from epicat.overlay import region_boxes
from epicat.subs import Cue, read_srt, write_srt
from epicat.translate import _parse_numbered, word_budgets


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
        with self.assertRaises(KeyError):
            Config().apply_overrides({"band.nope": 1})


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


if __name__ == "__main__":
    unittest.main()
