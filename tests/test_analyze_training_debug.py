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
            records.append(
                {
                    "epoch": epoch,
                    "train": {
                        "task/classification/loss/total": _metric(train_loss),
                        "optimization/gradient_clip_fraction": _metric(0.5),
                        "optimization/gradient_total_preclip": _metric(2.0),
                        "optimization/amp_overflow": _metric(0.0),
                    },
                    "validation": {
                        "task/classification/loss/total": _metric(validation_loss)
                    },
                    "epoch_metrics": {"step_success_fraction": 1.0},
                    "updates": {},
                }
            )
        result = analyze(
            {
                "epochs": records,
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


if __name__ == "__main__":
    unittest.main()
