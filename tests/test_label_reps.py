import csv
import tempfile
import unittest
from pathlib import Path

from scripts.label_reps import FIELDS, LABELS, ZONES, already_labelled, append_row


class LabelRepsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "labels.csv"

    def tearDown(self):
        self.directory.cleanup()

    def row(self, rep_id, label="3", zone="5"):
        return {
            "rep_id": rep_id,
            "video": "data/clip.mp4",
            "contact_frame": 120,
            "label": label,
            "zone": zone,
            "labeler": "will",
            "notes": "",
        }

    def test_header_written_once_and_rows_append(self):
        append_row(self.path, self.row("a"))
        append_row(self.path, self.row("b", "1", zone="6"))

        with self.path.open() as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual([r["rep_id"] for r in rows], ["a", "b"])
        self.assertEqual([r["label"] for r in rows], ["3", "1"])
        self.assertEqual([r["zone"] for r in rows], ["5", "6"])
        self.assertEqual(self.path.read_text().count(",".join(FIELDS)), 1)

    def test_the_scale_is_the_one_coaches_use(self):
        self.assertEqual(sorted(LABELS.values()), ["0", "1", "2", "3"])
        self.assertEqual(sorted(ZONES.values()), ["1", "2", "3", "4", "5", "6"])

    def test_zone_may_be_left_blank(self):
        append_row(self.path, self.row("a", zone=""))

        with self.path.open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["zone"], "")

    def test_resuming_skips_reps_already_labelled(self):
        append_row(self.path, self.row("a"))
        append_row(self.path, self.row("b"))

        self.assertEqual(already_labelled(self.path), {"a", "b"})

    def test_missing_file_is_an_empty_set_not_a_crash(self):
        self.assertEqual(already_labelled(self.path), set())


if __name__ == "__main__":
    unittest.main()
