import unittest

import torch
import torch.nn as nn

from evaluate_fmv2_paper_protocol import _encode, _paper_tasks


class _HistoryCheckingEmbedder(nn.Module):
    def forward(self, events, use_time_adapter=False, time_scale_factor=None):
        rows = []
        for event in events:
            history = event.get("history_features", (0.0, 0.0, 0.0, 0.0))
            if not isinstance(history, tuple) or len(history) != 4:
                raise TypeError("historical-memory schema was materialized")
            rows.append(history)
        return torch.as_tensor(rows, dtype=torch.float32)


class _FirstUnmaskedEncoder(nn.Module):
    def forward(self, values, src_key_padding_mask=None, task_type=None):
        return values[:, 0]


class _FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.embedder = _HistoryCheckingEmbedder()
        self.encoder = _FirstUnmaskedEncoder()
        self.pad_event = {"activity_name": "", "case_id": "pad"}

    def adapt_task_embeddings(self, values, task_type):
        return values


class PaperProtocolEncodingTests(unittest.TestCase):
    def test_paper_cohort_requires_two_observed_events(self):
        trace = [
            {
                "case_id": "case-a",
                "activity_id": index,
                "timestamp": float(index),
            }
            for index in range(4)
        ]
        tasks = _paper_tasks([trace], "classification")
        self.assertEqual(len(tasks), 2)
        self.assertEqual([len(task[0]) for task in tasks], [2, 3])

    def test_sparse_historical_memory_records_keep_dictionary_schema(self):
        tasks = [
            (
                [
                    {"activity_name": "history", "history_features": (2.0, 1.0, 0.5, 3.0)},
                    {"activity_name": "current"},
                ],
                0,
                "case-a",
            ),
            ([{"activity_name": "short"}], 1, "case-b"),
        ]
        normalized, retrieval = _encode(
            [_FakeExpert(), _FakeExpert()], tasks, "classification", batch_size=2
        )
        self.assertEqual([tuple(values.shape) for values in normalized], [(2, 4), (2, 4)])
        self.assertEqual(tuple(retrieval.shape), (2, 4))
        self.assertTrue(torch.isfinite(retrieval).all())


if __name__ == "__main__":
    unittest.main()
