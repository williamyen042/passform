import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import annotate


class DatasetTest(unittest.TestCase):
    """The corrections are the part worth testing: labelling mistakes are
    expected, and a correction that quietly loses a row is worse than the
    mistake it fixes."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.source = root / "practice_01.mp4"
        self.source.write_bytes(b"not really a video")
        self.cut = []
        # ffmpeg is exercised by using the tool; here only the bookkeeping is.
        annotate.extract_clip = lambda src, dst, start, end: (
            self.cut.append((dst.name, start, end)), dst.write_bytes(b"clip"))
        self.data = annotate.Dataset(root / "volleyball_dataset")

    def tearDown(self):
        self.directory.cleanup()

    def rows(self):
        with self.data.csv_path.open() as handle:
            return list(csv.DictReader(handle))

    def add(self, start=10.0, end=17.12, quality=3, position=6):
        return self.data.add(self.source, start, end, quality, position)

    def test_a_saved_rep_lands_in_the_csv_with_its_clip(self):
        rep = self.add()

        self.assertEqual(rep["filename"], "rep_0001.mp4")
        self.assertTrue((self.data.clips / "rep_0001.mp4").exists())
        self.assertEqual(self.rows(), [{
            "rep_id": "1", "filename": "rep_0001.mp4",
            "source_video": "practice_01.mp4", "start_time": "10.0",
            "end_time": "17.12", "duration": "7.12", "quality": "3",
            "position": "6",
        }])

    def test_ids_and_filenames_never_repeat(self):
        self.add()
        self.add(20.0, 27.0, 2, 5)
        self.assertEqual([row["filename"] for row in self.rows()],
                         ["rep_0001.mp4", "rep_0002.mp4"])

    def test_relabelling_does_not_recut_the_clip(self):
        rep = self.add()
        self.cut.clear()

        updated = self.data.update(rep["rep_id"], {"quality": 2, "position": 5})

        self.assertEqual(self.cut, [])
        self.assertEqual((updated["quality"], updated["position"]), (2, 5))
        self.assertEqual(updated["history"][0]["was"]["quality"], 3)

    def test_retiming_recuts_the_clip_and_keeps_the_old_times(self):
        rep = self.add()
        self.cut.clear()

        updated = self.data.update(rep["rep_id"], {"start_time": 11.5})

        self.assertEqual(self.cut, [("rep_0001.mp4", 11.5, 17.12)])
        self.assertEqual(updated["duration"], 5.62)
        self.assertEqual(updated["history"][0]["was"]["start_time"], 10.0)
        self.assertTrue(updated["history"][0]["re_extracted"])

    def test_undo_leaves_the_csv_clean_and_the_metadata_complete(self):
        rep = self.add()
        self.add(20.0, 27.0, 2, 5)

        self.data.update(rep["rep_id"], {"deleted": True})

        self.assertEqual([row["rep_id"] for row in self.rows()], ["2"])
        meta = json.loads(self.data.meta_path.read_text())
        self.assertEqual(len(meta["reps"]), 2)
        self.assertTrue(meta["reps"][0]["deleted"])

    def test_the_scale_is_written_into_the_dataset(self):
        meta = json.loads(self.data.meta_path.read_text())
        self.assertEqual(sorted(meta["quality_labels"]), ["0", "1", "2", "3"])
        self.assertEqual(sorted(meta["position_labels"]),
                         ["1", "2", "3", "4", "5", "6"])

    def test_bad_reps_are_refused_before_ffmpeg_runs(self):
        for bad in [(17.0, 10.0, 3, 6), (10.0, 10.1, 3, 6),
                    (10.0, 17.0, 4, 6), (10.0, 17.0, 3, 0)]:
            with self.assertRaises(ValueError):
                self.add(*bad)
        self.assertEqual(self.cut, [])


class RangeTest(unittest.TestCase):
    """Seeking a <video> is all range requests, and http.server has none."""

    def test_ranges(self):
        self.assertIsNone(annotate.parse_range(None, 100))
        self.assertIsNone(annotate.parse_range("bytes=--", 100))
        self.assertEqual(annotate.parse_range("bytes=0-", 100), (0, 99))
        self.assertEqual(annotate.parse_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(annotate.parse_range("bytes=10-999", 100), (10, 99))
        self.assertEqual(annotate.parse_range("bytes=-20", 100), (80, 99))
        self.assertIsNone(annotate.parse_range("bytes=200-300", 100))


if __name__ == "__main__":
    unittest.main()


class VballDecodeTest(unittest.TestCase):
    """The heatmap decode is the whole detector: everything else is plumbing."""

    def setUp(self):
        import numpy as np
        from core import vball_detector
        self.np = np
        self.mod = vball_detector

    def heatmap(self, x, y, peak=0.9):
        blob = self.np.zeros((self.mod.INPUT_HEIGHT, self.mod.INPUT_WIDTH),
                             dtype=self.np.float32)
        blob[y - 3:y + 4, x - 3:x + 4] = peak
        return blob

    def test_finds_the_centre_of_the_blob(self):
        point = self.mod._decode_heatmap(self.heatmap(256, 144), 0.5)
        self.assertIsNotNone(point)
        x, y, peak = point
        self.assertAlmostEqual(x, 256 / self.mod.INPUT_WIDTH, places=2)
        self.assertAlmostEqual(y, 144 / self.mod.INPUT_HEIGHT, places=2)
        self.assertAlmostEqual(peak, 0.9, places=5)

    def test_takes_the_largest_blob_when_two_fire(self):
        both = self.np.maximum(self.heatmap(100, 100), self.heatmap(400, 200))
        both[195:206, 395:406] = 0.9          # make the second blob bigger
        x, _, _ = self.mod._decode_heatmap(both, 0.5)
        self.assertGreater(x, 0.5)

    def test_nothing_above_threshold_is_not_a_detection(self):
        self.assertIsNone(self.mod._decode_heatmap(self.heatmap(256, 144, 0.2), 0.5))
