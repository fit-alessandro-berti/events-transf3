import unittest

from analyze_training_debug import analyze


def _metric(value):
    return {"count": 1, "mean": value, "std": 0.0, "min": value, "max": value}


class AnalyzeTrainingDebugTests(unittest.TestCase):
    def test_analysis_detects_front_loaded_overfitting_and_clipping(self):
        records = []
        train = [1.0, 0.8, 0.6, 0.5]
        validation = [1.0, 0.7, 0.8, 0.9]
        for epoch, (train_loss, validation_loss) in enumerate(
            zip(train, validation), start=1
        ):
            train_accuracy = [0.2, 0.4, 0.5, 0.6][epoch - 1]
            validation_accuracy = [0.2, 0.5, 0.4, 0.3][epoch - 1]
            records.append(
                {
                    "epoch": epoch,
                    "train": {
                        "task/classification/loss/total": _metric(train_loss),
                        "task/classification/head/classification/nll": _metric(
                            train_loss
                        ),
                        "task/classification/head/classification/accuracy": _metric(
                            train_accuracy
                        ),
                        "optimization/gradient_clip_fraction": _metric(0.5),
                        "optimization/gradient_total_preclip": _metric(2.0),
                        "optimization/amp_overflow": _metric(0.0),
                    },
                    "validation": {
                        "task/classification/loss/total": _metric(validation_loss),
                        "task/classification/head/classification/nll": _metric(
                            validation_loss
                        ),
                        "task/classification/head/classification/accuracy": _metric(
                            validation_accuracy
                        ),
                    },
                    "epoch_metrics": {"step_success_fraction": 1.0},
                    "updates": {},
                }
            )
        result = analyze(
            {
                "epochs": records,
                "configuration": {
                    "overfitting_patience": 2,
                    "overfitting_relative_tolerance": 0.02,
                },
                "generalization": {
                    "classification": {"overfitting_signal": True}
                },
            }
        )
        kinds = {finding["kind"] for finding in result["findings"]}
        self.assertIn("overfitting", kinds)
        self.assertIn("frequent_gradient_clipping", kinds)
        self.assertEqual(
            result["tasks"]["classification"]["validation_loss"]["best_epoch"],
            2,
        )
        self.assertTrue(
            result["tasks"]["classification"]["invariant_overfitting"][
                "overfitting_signal"
            ]
        )
        self.assertTrue(
            result["tasks"]["classification"]["invariant_overfitting"][
                "generalization_gap_signal"
            ]
        )
        self.assertTrue(
            result["tasks"]["classification"]["decision_overfitting"][
                "overfitting_signal"
            ]
        )

    def test_analysis_reports_pool_and_stagnant_auxiliary(self):
        records = []
        for epoch in range(1, 4):
            records.append(
                {
                    "epoch": epoch,
                    "schedule": {"retrieval_k": 10},
                    "train": {
                        "task/classification/loss/total": _metric(2.0),
                        "task/classification/loss/primary": _metric(1.3),
                        "task/classification/loss/routing_weighted": _metric(0.7),
                    },
                    "validation": {
                        "task/classification/loss/total": _metric(2.1),
                        "task/classification/pool/4/loss/total": _metric(3.0),
                        "task/classification/pool/4/head/classification/accuracy": (
                            _metric(0.2)
                        ),
                    },
                    "epoch_metrics": {},
                    "updates": {},
                }
            )
        result = analyze({"epochs": records}, {4: "source.xes.gz"})
        kinds = {finding["kind"] for finding in result["findings"]}
        self.assertIn("large_stagnant_auxiliary_loss", kinds)
        self.assertEqual(result["pools"]["classification"][0]["pool"], 4)
        self.assertEqual(
            result["pools"]["classification"][0]["log"], "source.xes.gz"
        )


if __name__ == "__main__":
    unittest.main()
