import csv
import tempfile
import unittest
from pathlib import Path

from compare_fmv3_results import PAIR_FIELDS, compare


class CompareFMV3ResultsTests(unittest.TestCase):
    def _write(self, path, rows):
        fields = [*PAIR_FIELDS, "balanced_accuracy", "mae_hours", "r2"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, task, repetition, **metrics):
        row = {field: "fixed" for field in PAIR_FIELDS}
        row.update({"task": task, "repetition": str(repetition), **metrics})
        return row

    def test_compare_requires_and_summarizes_paired_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left.csv", root / "right.csv"
            self._write(
                left,
                [
                    self._row("classification", 0, balanced_accuracy="0.4"),
                    self._row("classification", 1, balanced_accuracy="0.5"),
                    self._row("regression", 0, mae_hours="10", r2="0.1"),
                ],
            )
            self._write(
                right,
                [
                    self._row("classification", 0, balanced_accuracy="0.5"),
                    self._row("classification", 1, balanced_accuracy="0.5"),
                    self._row("regression", 0, mae_hours="8", r2="0.3"),
                ],
            )
            result = compare(left, right)
            classification = result["tasks"]["classification"]["metrics"]
            self.assertAlmostEqual(
                classification["balanced_accuracy"]["candidate_minus_reference"],
                0.05,
            )
            self.assertEqual(
                classification["balanced_accuracy"]["candidate_wins"], 1
            )
            self.assertAlmostEqual(
                result["tasks"]["classification"]["by_log"]["fixed"][
                    "balanced_accuracy"
                ]["candidate_minus_reference"],
                0.05,
            )
            regression = result["tasks"]["regression"]["metrics"]["mae_hours"]
            self.assertEqual(regression["candidate_wins"], 1)
            self.assertEqual(regression["candidate_minus_reference"], -2.0)
            r2 = result["tasks"]["regression"]["metrics"]["r2"]
            self.assertEqual(r2["candidate_wins"], 1)
            self.assertAlmostEqual(r2["candidate_minus_reference"], 0.2)

    def test_compare_rejects_unpaired_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left.csv", root / "right.csv"
            self._write(left, [self._row("classification", 0)])
            self._write(right, [self._row("classification", 1)])
            with self.assertRaisesRegex(ValueError, "not paired"):
                compare(left, right)


if __name__ == "__main__":
    unittest.main()
