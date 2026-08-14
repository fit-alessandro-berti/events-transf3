import unittest

from compare_metric_objectives import compare


def _metric(value):
    return {"count": 1, "mean": value, "std": 0.0, "min": value, "max": value}


def _run(class_profile="equilibrated", regression_profile="equilibrated", gain=0.0):
    epochs = []
    for epoch in (1, 2):
        improvement = gain * (epoch - 1)
        epochs.append({
            "epoch": epoch,
            "validation": {
                "task/classification/head/classification/episode_accuracy": _metric(0.4 + improvement),
                "task/classification/head/classification/episode_balanced_accuracy": _metric(0.3 + improvement),
                "task/classification/head/classification/episode_macro_f1": _metric(0.25 + improvement),
                "task/classification/head/classification/nll": _metric(2.0 - improvement),
                "task/classification/loss/classification/brier_surrogate_raw": _metric(0.3 - improvement / 2),
                "task/regression/head/regression/mae_hours": _metric(100.0 - 10 * improvement),
                "task/regression/head/regression/rmse_hours": _metric(150.0 - 10 * improvement),
                "task/regression/head/regression/r2": _metric(0.1 + improvement),
            },
        })
    return (
        {"epochs": epochs},
        {
            "classification_objective": {"profile": class_profile},
            "fmv3_head": {"regression_objective_profile": regression_profile},
        },
    )


class CompareMetricObjectivesTests(unittest.TestCase):
    def test_profile_and_equilibrated_epoch_selection(self):
        result = compare(
            {
                "baseline": _run(),
                "accuracy": _run("accuracy", gain=0.1),
                "r2": _run(regression_profile="r2", gain=0.1),
            },
            "baseline",
        )
        self.assertEqual(result["runs"]["accuracy"]["profile_selected_epoch"], 2)
        self.assertEqual(result["runs"]["r2"]["profile_selected_epoch"], 2)
        self.assertEqual(
            result["runs"]["accuracy"]["profiles"]["classification"],
            "accuracy",
        )
        # Half-Brier is converted back to the conventional evaluation scale.
        self.assertAlmostEqual(
            result["runs"]["baseline"]["profile_selected_metrics"][
                "classification_brier"
            ],
            0.6,
        )

    def test_unknown_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown baseline"):
            compare({"run": _run()}, "missing")


if __name__ == "__main__":
    unittest.main()
