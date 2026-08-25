import random
import tempfile
import unittest
from pathlib import Path

from training_log_sets import (
    active_training_log_sets,
    choose_training_log_set,
    combined_training_log_paths,
    resolve_training_log_sets,
)


class TrainingLogSetTests(unittest.TestCase):
    def test_resolves_explicit_and_directory_sets_with_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "base.xes.gz"
            explicit.touch()
            generated = root / "generated"
            generated.mkdir()
            (generated / "one.xes").touch()
            (generated / "two.xes.gz").touch()
            (generated / "ignore.txt").touch()

            resolved = resolve_training_log_sets(
                {
                    "epochs": 12,
                    "training_log_sets": [
                        {
                            "name": "base",
                            "epochs": (1, 12),
                            "log_paths": {"base": str(explicit)},
                        },
                        {
                            "name": "generated",
                            "epochs": (10, 12),
                            "directory": str(generated),
                        },
                    ],
                }
            )

            self.assertEqual([item["name"] for item in resolved], ["base", "generated"])
            self.assertEqual(
                set(resolved[1]["log_paths"]), {"one", "two"}
            )
            self.assertEqual(
                [item["name"] for item in active_training_log_sets(resolved, 9)],
                ["base"],
            )
            self.assertEqual(
                [item["name"] for item in active_training_log_sets(resolved, 10)],
                ["base", "generated"],
            )
            self.assertEqual(
                set(combined_training_log_paths(resolved)),
                {"base/base", "generated/one", "generated/two"},
            )

    def test_choice_is_uniform_choice_from_active_sets(self):
        sets = [
            {"name": "early", "start_epoch": 1, "end_epoch": 10},
            {"name": "late", "start_epoch": 5, "end_epoch": 10},
        ]
        rng = random.Random(7)
        choices = {
            choose_training_log_set(sets, 6, rng)["name"] for _ in range(20)
        }
        self.assertEqual(choices, {"early", "late"})
        self.assertEqual(
            choose_training_log_set(sets, 2, rng)["name"], "early"
        )

    def test_rejects_uncovered_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.xes"
            path.touch()
            with self.assertRaisesRegex(ValueError, "epoch\\(s\\): 3"):
                resolve_training_log_sets(
                    {
                        "epochs": 3,
                        "training_log_sets": [
                            {
                                "name": "base",
                                "epochs": (1, 2),
                                "log_paths": {"base": str(path)},
                            }
                        ],
                    }
                )

    def test_legacy_training_paths_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.xes"
            path.touch()
            resolved = resolve_training_log_sets(
                {
                    "epochs": 3,
                    "log_paths": {"training": {"base": str(path)}},
                }
            )
            self.assertEqual(resolved[0]["name"], "default")
            self.assertEqual(resolved[0]["start_epoch"], 1)
            self.assertEqual(resolved[0]["end_epoch"], 3)

    def test_content_addressed_directory_rejects_unknown_or_changed_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.xes").write_bytes(b"one")
            base = {
                "epochs": 1,
                "training_log_sets": [
                    {"name": "snapshot", "directory": str(root)}
                ],
            }
            resolved = resolve_training_log_sets(base)
            manifest_hash = resolved[0]["manifest_sha256"]
            protected = {
                "epochs": 1,
                "training_log_sets": [
                    {
                        "name": "snapshot",
                        "directory": str(root),
                        "manifest_sha256": manifest_hash,
                        "require_manifest": True,
                        "allow_aggregate_manifest_only": True,
                    }
                ],
            }
            resolve_training_log_sets(protected)
            (root / "two.xes").write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                resolve_training_log_sets(protected)

    def test_per_file_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.xes"
            path.write_bytes(b"actual")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                resolve_training_log_sets(
                    {
                        "epochs": 1,
                        "training_log_sets": [
                            {
                                "name": "source",
                                "log_paths": {"base": str(path)},
                                "file_sha256": {"base": "0" * 64},
                            }
                        ],
                    }
                )


if __name__ == "__main__":
    unittest.main()
