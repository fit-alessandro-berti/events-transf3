import copy
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from components.learned_event_embedder import LearnedEventEmbedder
from components.char_cnn_embedder import CharCNNEmbedder
from components.moe_model import MoEModel
from components.prototypical_head import PrototypicalHead
from config import CONFIG
from config_utils import (
    apply_experiment_config,
    validate_exact_resume_config,
    validate_run_configuration,
)
import data_generator
from data_generator import XESLogLoader
from evaluation.fmv3_metrics import regression_metrics
from training import adaptive_clip_grad_, build_training_state
from training_strategies.retrieval_strategy import (
    run_retrieval_step,
    select_mixed_negative_indices,
)
from training_log_sets import (
    resolve_training_log_sets,
    training_log_set_weight,
    validate_evaluation_split,
    validate_training_evaluation_disjointness,
)
from utils.data_utils import (
    TaskPoolView,
    create_episode,
    get_classification_and_regression_tasks,
    prefix_task_length,
)


class FoundationImprovementTests(unittest.TestCase):
    def test_retrain_and_clip_configs_resolve_to_real_global_clip_training(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "training_debug_full_retrain.yaml": 1.0,
            "training_debug_clip5_retrain.yaml": 5.0,
            "training_debug_clip10_retrain.yaml": 10.0,
            "training_debug_head_focused_retrain.yaml": 1.0,
        }
        for filename, clip_norm in expected.items():
            config = copy.deepcopy(CONFIG)
            apply_experiment_config(config, str(root / "configs/fmv3" / filename))
            self.assertEqual(validate_run_configuration(config), "train")
            self.assertTrue(config["training_enabled"])
            self.assertEqual(config["trainable_scope"], "all")
            self.assertNotIn("assembly", config)
            self.assertEqual(config["gradient_clip_mode"], "global")
            self.assertEqual(config["gradient_clip_norm"], clip_norm)

    def test_initialize_requires_explicit_artifacts_and_epoch_semantics(self):
        base = {
            "run_mode": "initialize",
            "training_enabled": True,
            "initialize_from_checkpoint": "model_epoch_4.pth",
            "gradient_clip_mode": "global",
            "gradient_clip_norm": 5.0,
        }
        with self.assertRaisesRegex(ValueError, "initialize_from_artifacts"):
            validate_run_configuration(base)
        valid = {
            **base,
            "initialize_from_artifacts": "training_artifacts.pth",
            "source_epoch": 4,
            "additional_epochs": 2,
            "epochs": 6,
            "reset_optimizer": True,
            "reset_scheduler": True,
        }
        self.assertEqual(validate_run_configuration(valid), "initialize")
        with self.assertRaisesRegex(ValueError, r"source_epoch \+ additional_epochs"):
            validate_run_configuration({**valid, "epochs": 7})

    def test_default_corpus_includes_every_log_and_replays_source(self):
        log_sets = resolve_training_log_sets(CONFIG)
        self.assertEqual([item["name"] for item in log_sets], ["source", "synthetic"])
        repository_root = Path(__file__).resolve().parents[1]
        logs_directory = repository_root / "logs"
        disk_log_paths = {
            path.resolve()
            for pattern in ("*.xes", "*.xes.gz")
            for path in logs_directory.rglob(pattern)
        }
        configured_training_paths = {
            (repository_root / path).resolve()
            for item in log_sets
            for path in item["log_paths"].values()
        }
        self.assertEqual(disk_log_paths, configured_training_paths)
        self.assertTrue(
            any(
                Path(path).name == "00013_clos2rep.xes.gz"
                for item in log_sets
                for path in item["log_paths"].values()
            )
        )
        self.assertNotIn("meta_test", CONFIG["evaluation_log_sets"])
        self.assertEqual(training_log_set_weight(log_sets[0], 1), 1.00)
        self.assertEqual(training_log_set_weight(log_sets[1], 1), 0.00)
        self.assertEqual(training_log_set_weight(log_sets[0], 10), 1.00)
        self.assertEqual(training_log_set_weight(log_sets[1], 10), 0.00)
        self.assertEqual(training_log_set_weight(log_sets[0], 11), 0.70)
        self.assertEqual(training_log_set_weight(log_sets[1], 11), 0.30)
        self.assertEqual(training_log_set_weight(log_sets[0], 40), 0.70)
        self.assertEqual(training_log_set_weight(log_sets[1], 40), 0.30)
        validate_training_evaluation_disjointness(CONFIG, log_sets)

    def test_disjointness_rejects_copied_or_renamed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.xes"
            copied = root / "renamed.xes"
            train.write_bytes(b"same-xes-content")
            shutil.copy2(train, copied)
            config = {
                "epochs": 1,
                "training_log_sets": [
                    {"name": "train", "log_paths": {"train": str(train)}}
                ],
                "log_paths": {"testing": {"test": str(copied)}},
            }
            log_sets = resolve_training_log_sets(config)
            with self.assertRaisesRegex(ValueError, "content overlap"):
                validate_training_evaluation_disjointness(config, log_sets)

    def test_screening_and_meta_test_splits_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen = root / "screen.xes"
            meta = root / "meta.xes"
            screen.touch()
            meta.touch()
            config = {
                "evaluation_log_sets": {
                    "screening": {"screen": str(screen)},
                    "meta_test": {"meta": str(meta)},
                }
            }
            validate_evaluation_split(config, {"meta": str(meta)}, "meta_test")
            with self.assertRaisesRegex(ValueError, "not in the 'screening' split"):
                validate_evaluation_split(config, {"meta": str(meta)}, "screening")

    def test_lazy_joint_tasks_include_prefix_length_one(self):
        trace = [
            {"case_id": "c", "activity_id": index, "timestamp": float(index)}
            for index in range(4)
        ]
        classification, regression = get_classification_and_regression_tasks(
            [trace], max_seq_len=2, minimum_prefix_length=1
        )
        self.assertIsInstance(classification, TaskPoolView)
        self.assertIs(classification.records, regression.records)
        self.assertEqual([len(item[0]) for item in classification], [1, 2, 2])
        self.assertEqual([item[1] for item in classification], [1, 2, 3])

    def test_long_prefix_adds_compressed_historical_memory(self):
        trace = [
            {"case_id": "c", "activity_id": index % 2, "timestamp": float(index)}
            for index in range(5)
        ]
        classification, _ = get_classification_and_regression_tasks(
            [trace],
            config={
                "data": {
                    "max_sequence_length": 2,
                    "minimum_prefix_length": 1,
                    "historical_memory_enabled": True,
                }
            },
        )
        prefix = classification[-1][0]
        self.assertEqual(len(prefix), 3)
        self.assertEqual(prefix[0]["is_history_summary"], 1.0)
        self.assertEqual(prefix[0]["history_features"][:2], (2.0, 2.0))
        self.assertEqual([event["activity_id"] for event in prefix[1:]], [0, 1])
        self.assertEqual(prefix_task_length(classification, -1), 4)

    def test_historical_memory_distinguishes_order_with_same_counts(self):
        def final_prefix(activity_ids):
            trace = [
                {
                    "case_id": "c",
                    "activity_id": activity_id,
                    "timestamp": float(index),
                }
                for index, activity_id in enumerate(activity_ids)
            ]
            classification, _ = get_classification_and_regression_tasks(
                [trace],
                config={
                    "data": {
                        "max_sequence_length": 2,
                        "minimum_prefix_length": 1,
                        "historical_memory_enabled": True,
                    }
                },
            )
            return classification[-1][0][0]

        left = final_prefix([0, 1, 2, 1, 1, 3])
        right = final_prefix([0, 2, 1, 1, 1, 3])
        self.assertEqual(left["history_features"], right["history_features"])
        self.assertNotEqual(
            left["history_transition_features"],
            right["history_transition_features"],
        )

    def test_duration_scale_metrics_use_full_case_duration_when_supplied(self):
        metrics = regression_metrics(
            [0.5, 0.5], [0.25, 1.0], process_duration_hours=[0.75, 200.0]
        )
        self.assertEqual(metrics["duration_scale_basis"], "full_case_duration")
        self.assertEqual(metrics["duration_scale_metrics"]["<1h"]["n"], 1)
        self.assertEqual(metrics["duration_scale_metrics"][">=7d"]["n"], 1)

    def test_episodic_support_and_query_cases_are_disjoint(self):
        tasks = []
        for case in range(8):
            for label in (0, 1):
                tasks.append(([{"case_id": f"c{case}"}], label, f"c{case}"))
        episode = create_episode(
            tasks, (1, 1), 1, num_ways_range=(2, 2), shuffle_labels=False
        )
        self.assertIsNotNone(episode)
        support_cases = {item[0][-1]["case_id"] for item in episode[0]}
        query_cases = {item[0][-1]["case_id"] for item in episode[1]}
        self.assertFalse(support_cases & query_cases)

    def test_equal_timestamps_preserve_source_order(self):
        loader = XESLogLoader("learned")
        loader.training_activity_names = ["A", "B", "C"]
        loader.char_to_id = {"<PAD>": 0, "<UNK>": 1, "A": 2, "B": 3, "C": 4}
        frame = pd.DataFrame(
            {
                "case:concept:name": ["c", "c", "c"],
                "concept:name": ["B", "A", "C"],
                "time:timestamp": pd.to_datetime(
                    ["2024-01-01 09:00Z", "2024-01-01 09:00Z", "2024-01-01 10:00Z"]
                ),
                "org:resource": ["R", "R", "R"],
                "amount": [0.0, 0.0, 0.0],
            }
        )
        trace = loader.transform_dataframes({"test": frame})["test"][0]
        self.assertEqual([event["activity_name"] for event in trace], ["B", "A", "C"])

    def test_transform_is_log_local_before_attribute_truncation(self):
        loader = XESLogLoader("learned", max_generic_attributes=1)
        loader.training_activity_names = ["A"]
        loader.char_to_id = {"<PAD>": 0, "<UNK>": 1, "A": 2, "R": 3}
        common = {
            "case:concept:name": ["c"],
            "concept:name": ["A"],
            "time:timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"]),
            "org:resource": ["R"],
            "amount": [0.0],
        }
        primary = pd.DataFrame({**common, "z_valid": [7.0]})
        unrelated = pd.DataFrame(
            {**common, **{f"a{index:02d}": [index] for index in range(20)}}
        )
        alone = loader.transform_dataframes({"primary": primary})["primary"]
        together = loader.transform_dataframes(
            {"primary": primary, "unrelated": unrelated}
        )["primary"]
        self.assertEqual(
            alone[0][0]["generic_attributes"],
            together[0][0]["generic_attributes"],
        )
        # lifecycle plus the one valid generic attribute
        self.assertEqual(len(together[0][0]["generic_attributes"]), 2)

    def test_lifecycle_values_are_fitted_into_character_vocabulary(self):
        loader = XESLogLoader("learned")
        frame = pd.DataFrame(
            {
                "concept:name": ["A"],
                "org:resource": ["R"],
                "lifecycle:transition": ["ø-state"],
            }
        )
        with mock.patch.object(data_generator.pm4py, "read_xes", return_value=frame), mock.patch.object(
            data_generator.os.path, "exists", return_value=True
        ):
            loader.fit({"train": "unused.xes"})
        self.assertIn("ø", loader.char_to_id)

    def test_artifact_metadata_is_validated_instead_of_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.pth"
            loader = XESLogLoader("learned")
            loader.training_activity_names = ["A"]
            loader.activity_to_id = {"A": 0}
            loader.char_to_id = {"<PAD>": 0, "<UNK>": 1, "A": 2}
            metadata = {
                "training_log_manifest_sha256": "corpus",
                "preprocessing_sha256": "preprocessing",
                "model_architecture_sha256": "architecture",
            }
            loader.save_training_artifacts(path, metadata=metadata)
            restored = XESLogLoader("learned")
            restored.load_training_artifacts(path, expected_metadata=metadata)
            self.assertEqual(restored.char_to_id, loader.char_to_id)
            with self.assertRaisesRegex(ValueError, "preprocessing_sha256"):
                restored.load_training_artifacts(
                    path,
                    expected_metadata={
                        **metadata,
                        "preprocessing_sha256": "different",
                    },
                )

    def test_pretrained_artifacts_persist_maps_and_model_revision(self):
        class DummySentenceTransformer:
            def __init__(self, name, **kwargs):
                self.name = name
                self.revision = kwargs.get("revision")

            def get_sentence_embedding_dimension(self):
                return 2

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            data_generator, "SentenceTransformer", DummySentenceTransformer
        ):
            path = Path(directory) / "pretrained.pth"
            loader = XESLogLoader(
                "pretrained", sbert_model_name="model", sbert_model_revision="abc123"
            )
            loader.training_activity_names = ["A"]
            loader.activity_to_id = {"A": 0}
            loader.training_activity_embeddings = np.asarray([[1.0, 0.0]])
            loader.activity_embedding_map = {"A": np.asarray([1.0, 0.0])}
            loader.resource_embedding_map = {"R": np.asarray([0.0, 1.0])}
            loader.save_training_artifacts(path, metadata={"corpus": "hash"})

            restored = XESLogLoader(
                "pretrained", sbert_model_name="model", sbert_model_revision="abc123"
            )
            restored.load_training_artifacts(
                path, expected_metadata={"corpus": "hash"}
            )
            np.testing.assert_array_equal(
                restored.activity_embedding_map["A"], np.asarray([1.0, 0.0])
            )
            np.testing.assert_array_equal(
                restored.resource_embedding_map["R"], np.asarray([0.0, 1.0])
            )

    def test_character_pooling_masks_padding_and_empty_strings(self):
        embedder = CharCNNEmbedder(4, 3, 6, max_word_len=8)
        output = embedder(["", "A"], {"<PAD>": 0, "<UNK>": 1, "A": 2})
        torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
        self.assertTrue(torch.isfinite(output[1]).all())

    def test_optional_agc_does_not_suppress_zero_initialized_heads(self):
        zero_head = torch.nn.Parameter(torch.zeros(2, 3))
        zero_head.grad = torch.ones_like(zero_head)
        adaptive_clip_grad_([zero_head], factor=0.02, epsilon=1e-3)
        torch.testing.assert_close(zero_head.grad, torch.ones_like(zero_head))

        mature = torch.nn.Parameter(torch.ones(2, 3))
        mature.grad = torch.full_like(mature, 100.0)
        adaptive_clip_grad_([mature], factor=0.1, epsilon=1e-3)
        expected_ceiling = 0.1 * mature.detach().norm(dim=1)
        self.assertTrue(
            torch.all(mature.grad.detach().norm(dim=1) <= expected_ceiling + 1e-6)
        )

    def test_negative_curriculum_changes_selected_support(self):
        angles = torch.linspace(0.0, 1.2, 10)
        candidates = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        query = torch.tensor([[1.0, 0.0]])
        eligible = torch.ones(10, dtype=torch.bool)
        nearest = select_mixed_negative_indices(
            query, candidates, eligible, 4, pool_factor=2, random_fraction=0.0
        )
        generator = torch.Generator().manual_seed(11)
        mixed = select_mixed_negative_indices(
            query,
            candidates,
            eligible,
            4,
            pool_factor=2,
            random_fraction=1.0,
            generator=generator,
        )
        self.assertEqual(nearest.tolist(), [0, 1, 2, 3])
        self.assertNotEqual(mixed.tolist(), nearest.tolist())
        self.assertEqual(len(set(mixed.tolist())), 4)

    def test_pretrained_semantics_do_not_define_target_class_ids(self):
        class DummySentenceModel:
            def encode(self, names, **kwargs):
                return np.asarray(
                    [[float(index + 1), 1.0] for index, _ in enumerate(names)],
                    dtype=np.float32,
                )

        loader = XESLogLoader("learned")
        loader.strategy = "pretrained"
        loader.sbert_model = DummySentenceModel()
        loader.training_activity_names = ["Training activity"]
        loader.training_activity_embeddings = np.asarray([[1.0, 1.0]], dtype=np.float32)
        loader.activity_to_id = {"Training activity": 0}
        loader.activity_embedding_map = {
            "Training activity": np.asarray([1.0, 1.0], dtype=np.float32)
        }
        loader.resource_embedding_map = {
            "Unknown": np.zeros(2, dtype=np.float32)
        }
        loader.pad_embedding = np.zeros(2, dtype=np.float32)
        frame = pd.DataFrame(
            {
                "concept:name": ["New B", "New A"],
                "org:resource": ["Unknown", "Unknown"],
            }
        )
        raw = [[
            {
                "activity": name,
                "resource": "Unknown",
                "cost": 0.0,
                "time_from_start": float(index),
                "time_from_previous": float(index),
                "timestamp": float(index),
                "case_id": "c",
                "resource_missing": 0.0,
                "cost_missing": 0.0,
                "lifecycle": "complete",
                "lifecycle_missing": 0.0,
                "calendar_features": (0.0,) * 5,
                "generic_attributes": (),
            }
            for index, name in enumerate(("New B", "New A"))
        ]]
        transformed = loader._transform_pretrained(
            frame, raw, "concept:name", "org:resource"
        )
        ids = [event["activity_id"] for event in transformed[0]]
        self.assertEqual(ids, [1, 0])
        self.assertEqual(len(set(ids)), 2)

    def test_signed_cost_is_not_clamped_to_zero(self):
        embedder = LearnedEventEmbedder(
            char_vocab_size=4,
            char_emb_dim=4,
            char_cnn_out_dim=6,
            num_feat_dim=3,
            d_model=8,
            dropout=0.0,
            max_string_length=8,
            attribute_hash_buckets=32,
        )
        embedder.char_to_id = {"<PAD>": 0, "<UNK>": 1, "A": 2, "R": 3}
        captured = []
        hook = embedder.projection.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach())
        )
        base = {
            "activity_name": "A",
            "resource_name": "R",
            "time_from_start": 1.0,
            "time_from_previous": 1.0,
        }
        embedder([{**base, "cost": -3.0}, {**base, "cost": 3.0}])
        hook.remove()
        cost_features = captured[0][:, -3]
        self.assertLess(cost_features[0].item(), 0.0)
        self.assertAlmostEqual(
            cost_features[0].item(), -cost_features[1].item(), places=6
        )

    def test_residual_regressor_can_extrapolate_past_support(self):
        head = PrototypicalHead(
            feature_dim=2,
            regression_mode="raw_hours_knn",
            regression_residual_enabled=True,
            regression_residual_hidden_dim=4,
        )
        with torch.no_grad():
            head.regression_residual_head[-1].bias[:] = torch.tensor([4.0, -10.0])
        support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # Stored labels are sqrt(hours): the observed target range is [1, 4].
        labels = torch.tensor([1.0, 2.0])
        query = torch.tensor([[1.0, 1.0]])
        prediction, _, diagnostics = head.forward_regression(
            support, labels, query, return_diagnostics=True
        )
        self.assertGreater(prediction.item(), 4.0)
        self.assertIn("residual_anchor_gate", diagnostics)

    def test_vectorized_classification_matches_query_specific_reference(self):
        torch.manual_seed(3)
        pool = torch.randn(6, 5)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query = torch.randn(2, 5)
        local_positions = torch.tensor([[0, 2, 4], [1, 3, 5]])
        global_mask = torch.tensor(
            [
                [True, True, True, True, True, False],
                [True, False, True, True, True, True],
            ]
        )
        for mode in ("legacy_soft_knn", "coverage_fallback"):
            head = PrototypicalHead(classification_mode=mode)
            batched, classes, probabilities = head.forward_classification_batched(
                pool[local_positions],
                labels[local_positions],
                query,
                global_support_features=pool,
                global_support_labels=labels,
                global_support_mask=global_mask,
                candidate_classes=torch.tensor([0, 1, 2]),
            )
            self.assertEqual(classes.tolist(), [0, 1, 2])
            expected = []
            for row in range(2):
                logits, row_classes, row_probabilities = head.forward_classification(
                    pool[local_positions[row]],
                    labels[local_positions[row]],
                    query[row : row + 1],
                    global_support_features=pool[global_mask[row]],
                    global_support_labels=labels[global_mask[row]],
                    candidate_classes=torch.tensor([0, 1, 2]),
                )
                aligned = torch.zeros(3)
                aligned[row_classes.long()] = row_probabilities[0]
                expected.append(aligned)
            torch.testing.assert_close(
                probabilities, torch.stack(expected), atol=1e-6, rtol=1e-6
            )
            self.assertTrue(torch.isfinite(batched[torch.isfinite(batched)]).all())

    def test_small_pool_uses_adaptive_batch_instead_of_noop(self):
        class TinyRetrievalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Parameter(torch.randn(8, 5))
                self.proj_head = torch.nn.Identity()
                self.proto_head = PrototypicalHead(
                    classification_mode="legacy_soft_knn"
                )

            def _process_batch(self, sequences, **kwargs):
                indices = torch.tensor(
                    [sequence[0]["index"] for sequence in sequences],
                    dtype=torch.long,
                )
                return self.embeddings[indices]

        tasks = [
            ([{"index": index}], index % 2, f"case-{index // 2}")
            for index in range(8)
        ]
        config = {
            "retrieval_train_batch_size": 128,
            "retrieval_min_batch_size": 4,
            "retrieval_train_k": 4,
            "retrieval_min_per_class": 2,
            "retrieval_cls_pos_k": 1,
            "retrieval_contrastive_weight": 0.0,
            "retrieval_knn_aux_weight": 0.0,
            "retrieval_var_weight": 0.0,
            "retrieval_cov_weight": 0.0,
            "classification_separation_weight": 0.0,
            "classification_label_smoothing": 0.0,
            "classification_objective": {"profile": "accuracy", "weights": {}},
            "fmv3_training": {"episode_mix": {"balanced": 1.0}},
            "fmv3_head": {},
        }
        loss, _ = run_retrieval_step(
            TinyRetrievalModel(), tasks, "classification", config
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_shared_backbone_has_distinct_lightweight_experts(self):
        model = MoEModel(
            num_experts=2,
            strategy="learned",
            shared_backbone=True,
            num_feat_dim=3,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            proto_head_config={},
            char_vocab_size=4,
            char_embedding_dim=4,
            char_cnn_output_dim=6,
            max_string_length=8,
            attribute_hash_buckets=32,
            expert_adapter_enabled=True,
            expert_adapter_hidden_dim=4,
        )
        self.assertIs(model.experts[0].embedder, model.experts[1].embedder)
        self.assertIs(model.experts[0].encoder, model.experts[1].encoder)
        self.assertIsNot(
            model.experts[0].expert_adapter, model.experts[1].expert_adapter
        )

    def test_full_training_state_contains_exact_resume_inputs(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        state = build_training_state(
            1, model, optimizer, scheduler, scaler, random.Random(7), {"epochs": 2}
        )
        expected = {
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "log_set_rng",
            "config",
        }
        self.assertTrue(expected.issubset(state))

    def test_exact_resume_rejects_configuration_drift(self):
        saved = {"run_mode": "train", "epochs": 2, "lr": 1e-3}
        validate_exact_resume_config({**saved, "run_mode": "resume"}, saved)
        with self.assertRaisesRegex(ValueError, "lr"):
            validate_exact_resume_config(
                {"run_mode": "resume", "epochs": 2, "lr": 2e-3}, saved
            )

    def test_every_trainable_yaml_has_valid_schedule_and_checkpoint_semantics(self):
        root = Path(__file__).resolve().parents[1]
        checked = 0
        for path in sorted((root / "configs").rglob("*.yaml")):
            config = copy.deepcopy(CONFIG)
            apply_experiment_config(config, str(path))
            if not config.get("training_enabled", True):
                continue
            mode = validate_run_configuration(config)
            if mode != "assemble":
                resolve_training_log_sets(config)
            checked += 1
        self.assertGreater(checked, 0)

    def test_selected_endpoint_is_an_explicit_assembly_manifest(self):
        root = Path(__file__).resolve().parents[1]
        config = copy.deepcopy(CONFIG)
        apply_experiment_config(
            config, str(root / "configs/fmv3/example_selector_selected.yaml")
        )
        self.assertEqual(validate_run_configuration(config), "assemble")
        self.assertFalse(config["training_enabled"])
        self.assertEqual(
            set(config["assembly"]),
            {
                "base_checkpoint",
                "classification_checkpoint",
                "regression_checkpoint",
                "output_checkpoint",
                "artifacts",
            },
        )


if __name__ == "__main__":
    unittest.main()
