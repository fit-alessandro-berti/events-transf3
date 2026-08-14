import unittest

from analyze_training_debug import analyze


def _metric(value):
    return {"count": 1, "mean": value, "std": 0.0, "min": value, "max": value}


class AnalyzeTrainingDebugTests(unittest.TestCase):
    def test_analysis_distinguishes_transient_amp_warmup_overflow(self):
        records = []
        for epoch, overflow in enumerate([0.02, 0.0, 0.0], start=1):
            records.append(
                {
                    "epoch": epoch,
                    "train": {
                        "task/classification/loss/total": _metric(1.0),
                        "optimization/amp_overflow": _metric(overflow),
                    },
                    "validation": {
                        "task/classification/loss/total": _metric(1.0),
                    },
                    "epoch_metrics": {},
                    "updates": {},
                }
            )
        result = analyze({"epochs": records, "configuration": {}})
        findings = {finding["kind"]: finding for finding in result["findings"]}
        self.assertNotIn("amp_overflow", findings)
        self.assertEqual(findings["transient_amp_overflow"]["severity"], "low")

    def test_analysis_reports_persistent_amp_overflow(self):
        records = []
        for epoch in range(1, 4):
            records.append(
                {
                    "epoch": epoch,
                    "train": {
                        "task/classification/loss/total": _metric(1.0),
                        "optimization/amp_overflow": _metric(0.02),
                    },
                    "validation": {
                        "task/classification/loss/total": _metric(1.0),
                    },
                    "epoch_metrics": {},
                    "updates": {},
                }
            )
        result = analyze({"epochs": records, "configuration": {}})
        findings = {finding["kind"]: finding for finding in result["findings"]}
        self.assertEqual(findings["amp_overflow"]["severity"], "high")

    def test_analysis_flags_metric_gradient_imbalance(self):
        records = []
        for epoch in range(1, 4):
            records.append(
                {
                    "epoch": epoch,
                    "train": {
                        "task/regression/loss/total": _metric(1.0),
                        "task/regression/optimization/loss_gradient/primary/all/l2_norm": _metric(4.0),
                        "task/regression/optimization/loss_gradient/regression/median_ae_weighted/all/l2_norm": _metric(3.0),
                        "task/regression/optimization/loss_gradient/regression/mae_weighted/all/l2_norm": _metric(0.4),
                    },
                    "validation": {
                        "task/regression/loss/total": _metric(1.0),
                    },
                    "epoch_metrics": {},
                    "updates": {},
                }
            )
        result = analyze({"epochs": records, "configuration": {}})
        findings = [
            finding
            for finding in result["findings"]
            if finding["kind"] == "metric_gradient_imbalance"
        ]
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0]["metric"], "regression/median_ae_weighted"
        )
        self.assertAlmostEqual(
            findings[0]["evidence"]["largest_to_smallest_ratio"], 7.5
        )

    def test_analysis_flags_uniform_branch_gate_despite_mae_spread(self):
        records = []
        for epoch in range(1, 4):
            train = {"task/regression/loss/total": _metric(1.0)}
            validation = {
                "task/regression/loss/total": _metric(1.0),
                "task/regression/head/regression/branch_weight_entropy_mean": _metric(0.999),
            }
            for branch, mae in enumerate([10.0, 10.5, 11.0, 12.0]):
                validation[
                    f"task/regression/head/regression/branch_{branch}/mae_hours"
                ] = _metric(mae)
                validation[
                    f"task/regression/head/regression/branch_{branch}/weight_mean"
                ] = _metric(0.25)
            records.append(
                {
                    "epoch": epoch,
                    "train": train,
                    "validation": validation,
                    "epoch_metrics": {},
                    "updates": {},
                }
            )
        result = analyze({"epochs": records, "configuration": {}})
        findings = [
            finding
            for finding in result["findings"]
            if finding["kind"] == "near_uniform_regression_branch_mixture"
        ]
        self.assertEqual(len(findings), 1)
        self.assertAlmostEqual(findings[0]["evidence"]["relative_mae_spread"], 0.2)

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
        self.assertAlmostEqual(
            result["tasks"]["classification"]["validation_loss"]["epoch_three"][
                "fraction_of_best_improvement"
            ],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            result["tasks"]["classification"]["decision_overfitting"][
                "epoch_three"
            ]["fraction_of_best_improvement"],
            2.0 / 3.0,
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

    def test_analysis_flags_regression_error_concentrated_in_one_pool(self):
        validation = {"task/regression/loss/total": _metric(1.0)}
        for pool, mae in enumerate([10.0, 11.0, 200.0]):
            validation[
                f"task/regression/pool/{pool}/loss/total"
            ] = _metric(1.0)
            validation[
                f"task/regression/pool/{pool}/head/regression/mae_hours"
            ] = _metric(mae)
        records = [
            {
                "epoch": 1,
                "train": {"task/regression/loss/total": _metric(1.0)},
                "validation": validation,
                "epoch_metrics": {},
                "updates": {},
            }
        ]
        result = analyze(
            {"epochs": records, "configuration": {}},
            pool_names={0: "small-a", 1: "small-b", 2: "large"},
        )
        finding = next(
            finding
            for finding in result["findings"]
            if finding["kind"] == "regression_pool_error_concentration"
        )
        self.assertEqual(finding["evidence"]["worst_log"], "large")
        self.assertGreater(finding["evidence"]["ratio_to_median_pool"], 18.0)


if __name__ == "__main__":
    unittest.main()
