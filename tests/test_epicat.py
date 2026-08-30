"""Unit tests for the parts that do not need video files.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
