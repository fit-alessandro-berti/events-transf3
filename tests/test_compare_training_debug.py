import unittest

from compare_training_debug import compare


def _metric(value):
    return {"count": 1, "mean": value, "std": 0.0, "min": value, "max": value}


def _summary(class_nll, class_accuracy, confidence, regression_mae):
    epochs = []
    for epoch in range(1, 3):
        epochs.append(
            {
                "epoch": epoch,
                "train": {
                    "task/classification/head/classification/nll": _metric(
                        class_nll + 0.1
                    ),
                    "task/classification/head/classification/accuracy": _metric(
                        class_accuracy - 0.1
                    ),
                    "task/regression/head/regression/mae_hours": _metric(
                        regression_mae + 10
                    ),
                    "optimization/gradient_clip_fraction": _metric(0.5),
                    "optimization/gradient_total_preclip": _metric(2.0),
                    "optimization/amp_overflow": _metric(0.0),
                },
                "validation": {
                    "task/classification/head/classification/nll": _metric(
                        class_nll - 0.1 * epoch
                    ),
                    "task/classification/head/classification/accuracy": _metric(
                        class_accuracy + 0.1 * epoch
                    ),
                    "task/classification/head/classification/max_probability_mean": _metric(
                        confidence
                    ),
                    "task/regression/head/regression/mae_hours": _metric(
                        regression_mae - epoch
                    ),
                    "task/regression/head/regression/rmse_hours": _metric(
                        regression_mae * 2 - epoch
                    ),
                },
            }
        )
    return {"epochs": epochs}


class CompareTrainingDebugTests(unittest.TestCase):
    def test_compare_uses_invariant_metrics_and_reports_best_epochs(self):
        result = compare(
            {
                "baseline": _summary(2.0, 0.2, 0.4, 100.0),
                "candidate": _summary(1.8, 0.3, 0.35, 90.0),
            },
            "baseline",
        )
        candidate = result["runs"]["candidate"]
        self.assertEqual(candidate["best_classification_nll"]["epoch"], 2)
        self.assertEqual(candidate["best_regression_mae_hours"]["epoch"], 2)
        self.assertLess(
            candidate["best_joint_invariant_score"]["value"],
            result["runs"]["baseline"]["best_joint_invariant_score"]["value"],
        )
        self.assertAlmostEqual(candidate["last_confidence_gap"], -0.15)


if __name__ == "__main__":
    unittest.main()
