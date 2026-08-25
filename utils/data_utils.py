"""Compact prefix-task views and case-safe episode construction."""

from __future__ import annotations

import random
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from time_transf import transform_time


HISTORY_TRANSITION_BUCKETS = 16


def _history_transition_sketch(history, buckets=HISTORY_TRANSITION_BUCKETS):
    """Signed hash sketch retaining old activity identities and transition order."""
    values =[
        str(event.get("activity_name", event.get("activity_id", "<UNK>")))
        for event in history
    ]
    features =[f"activity:{value}"for value in values ]
    features .extend (
    f"transition:{left}->{right}"for left ,right in zip (values ,values [1 :]))
    sketch =[0.0 ]*int (buckets )
    for feature in features :
        digest =hashlib .blake2b (
        feature .encode ('utf-8',errors ='replace'),digest_size =8 ).digest ()
        raw =int .from_bytes (digest ,'little')
        bucket =raw %len (sketch )
        sign =1.0 if (raw >>8 )&1 else -1.0
        sketch [bucket]+=sign
    scale =math .sqrt (max (len (features ),1 ))
    return tuple (value /scale for value in sketch )


@dataclass(frozen=True, slots=True)
class PrefixTask:
    """One prefix location shared by classification and regression views."""

    trace: list
    end_index: int
    classification_target: int | None
    regression_target: float
    case_id: object
    max_sequence_length: int
    historical_memory_enabled: bool = False

    @property
    def prefix(self):
        start = max(0, self.end_index + 1 - self.max_sequence_length)
        retained = self.trace[start : self.end_index + 1]
        if not self.historical_memory_enabled or start == 0:
            return retained
        history = self.trace[:start]
        summary = dict(retained[0])
        if "activity_name" in summary:
            activity_width = len(summary.get("activity_char_ids", ()))
            resource_width = len(summary.get("resource_char_ids", ()))
            lifecycle_width = len(summary.get("lifecycle_char_ids", ()))
            summary["activity_name"] = ""
            summary["resource_name"] = ""
            summary["activity_char_ids"] = (0,) * activity_width
            summary["resource_char_ids"] = (0,) * resource_width
            summary["lifecycle_name"] = ""
            summary["lifecycle_char_ids"] = (0,) * lifecycle_width
        if "activity_embedding" in summary:
            summary["activity_embedding"] = summary["activity_embedding"] * 0
            summary["resource_embedding"] = summary["resource_embedding"] * 0
        activity_values = [event.get("activity_id") for event in history]
        unique_activities = len(set(activity_values))
        event_count = len(history)
        history_age = max(
            float(retained[0]["timestamp"] - history[0]["timestamp"]), 0.0
        )
        summary.update(
            {
                "activity_id": -100,
                "cost": 0.0,
                "time_from_start": 0.0,
                "time_from_previous": 0.0,
                "calendar_features": (0.0,) * 5,
                "resource_missing": 1.0,
                "cost_missing": 1.0,
                "lifecycle_missing": 1.0,
                "generic_attributes": (),
                "is_history_summary": 1.0,
                "history_features": (
                    float(event_count),
                    float(unique_activities),
                    float(max(event_count - unique_activities, 0))
                    / max(event_count, 1),
                    history_age,
                ),
                "history_transition_features": _history_transition_sketch(history),
            }
        )
        return [summary, *retained]


class TaskPoolView(Sequence):
    """Tuple-compatible lazy task view over shared :class:`PrefixTask` records."""

    def __init__(self, records, task_type, indices=None):
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unknown task type: {task_type}")
        self.records = records
        self.task_type = task_type
        self.indices = None if indices is None else tuple(indices)
        self._case_indices = None

    def __len__(self):
        return len(self.records) if self.indices is None else len(self.indices)

    def _tuple(self, record):
        target = (
            record.classification_target
            if self.task_type == "classification"
            else record.regression_target
        )
        return record.prefix, target, record.case_id

    def __getitem__(self, index):
        if isinstance(index, slice):
            positions = range(len(self))[index] if self.indices is None else self.indices[index]
            return [self._tuple(self.records[i]) for i in positions]
        record_index = index if self.indices is None else self.indices[index]
        return self._tuple(self.records[record_index])

    def __iter__(self) -> Iterator[tuple]:
        for index in (
            range(len(self.records)) if self.indices is None else self.indices
        ):
            yield self._tuple(self.records[index])

    @property
    def case_indices(self):
        if self._case_indices is None:
            grouped = defaultdict(list)
            record_indices = (
                range(len(self.records)) if self.indices is None else self.indices
            )
            for position, record_index in enumerate(record_indices):
                grouped[str(self.records[record_index].case_id)].append(position)
            self._case_indices = dict(grouped)
        return self._case_indices

    def subset(self, predicate: Callable[[tuple], bool]):
        selected = [
            record_index
            for record_index in (
                range(len(self.records)) if self.indices is None else self.indices
            )
            if predicate(self._tuple(self.records[record_index]))
        ]
        return TaskPoolView(self.records, self.task_type, selected)


def _data_setting(config, name, fallback):
    if not config:
        return fallback
    return (config.get("data", {}) or {}).get(name, fallback)


def get_classification_and_regression_tasks(
    log,
    max_seq_len=None,
    minimum_prefix_length=None,
    config=None,
):
    """Build two lazy task views in one pass over a transformed log."""
    max_seq_len = int(
        max_seq_len
        if max_seq_len is not None
        else _data_setting(config, "max_sequence_length", 10)
    )
    minimum_prefix_length = int(
        minimum_prefix_length
        if minimum_prefix_length is not None
        else _data_setting(config, "minimum_prefix_length", 1)
    )
    historical_memory_enabled = bool(
        _data_setting(config, "historical_memory_enabled", False)
    )
    if max_seq_len < 1:
        raise ValueError("max_sequence_length must be at least 1")
    if minimum_prefix_length < 1:
        raise ValueError("minimum_prefix_length must be at least 1")

    records = []
    for trace in log or ():
        if len(trace) < minimum_prefix_length + 1:
            continue
        case_id = trace[0]["case_id"]
        final_timestamp = trace[-1]["timestamp"]
        for end_index in range(minimum_prefix_length - 1, len(trace) - 1):
            next_activity = trace[end_index + 1].get("activity_id")
            remaining_hours = (
                final_timestamp - trace[end_index]["timestamp"]
            ) / 3600.0
            records.append(
                PrefixTask(
                    trace=trace,
                    end_index=end_index,
                    classification_target=next_activity,
                    regression_target=transform_time(remaining_hours),
                    case_id=case_id,
                    max_sequence_length=max_seq_len,
                    historical_memory_enabled=historical_memory_enabled,
                )
            )
    class_indices = [
        index
        for index, record in enumerate(records)
        if record.classification_target is not None
    ]
    return (
        TaskPoolView(
            records,
            "classification",
            None if len(class_indices) == len(records) else class_indices,
        ),
        TaskPoolView(records, "regression"),
    )


def get_task_data(
    log,
    task_type,
    max_seq_len=None,
    minimum_prefix_length=None,
    config=None,
):
    classification, regression = get_classification_and_regression_tasks(
        log,
        max_seq_len=max_seq_len,
        minimum_prefix_length=minimum_prefix_length,
        config=config,
    )
    if task_type == "classification":
        return classification
    if task_type == "regression":
        return regression
    raise ValueError(f"Unknown task type: {task_type}")


def prefix_task_length(task_pool, index):
    """Return the observed prefix length, excluding a synthetic history token."""
    if isinstance(task_pool, TaskPoolView):
        record_index = index if task_pool.indices is None else task_pool.indices[index]
        return task_pool.records[record_index].end_index + 1
    return len(task_pool[index][0])


def sample_task_batch(task_pool, batch_size, case_uniform_fraction=0.5):
    """Mix case-uniform and prefix-uniform samples without replacement."""
    batch_size = min(max(int(batch_size), 0), len(task_pool))
    if batch_size == 0:
        return []
    fraction = min(max(float(case_uniform_fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return random.sample(task_pool, batch_size)

    if isinstance(task_pool, TaskPoolView):
        by_case = task_pool.case_indices
    else:
        grouped = defaultdict(list)
        for position, item in enumerate(task_pool):
            grouped[str(item[2])].append(position)
        by_case = dict(grouped)

    target_case_uniform = min(batch_size, int(round(batch_size * fraction)))
    chosen_positions = set()
    cases = list(by_case)
    random.shuffle(cases)
    while len(chosen_positions) < target_case_uniform and cases:
        progressed = False
        for case in cases:
            candidates = [p for p in by_case[case] if p not in chosen_positions]
            if candidates:
                chosen_positions.add(random.choice(candidates))
                progressed = True
                if len(chosen_positions) >= target_case_uniform:
                    break
        if not progressed:
            break
        random.shuffle(cases)

    remaining = [p for p in range(len(task_pool)) if p not in chosen_positions]
    chosen_positions.update(
        random.sample(remaining, min(batch_size - len(chosen_positions), len(remaining)))
    )
    positions = list(chosen_positions)
    random.shuffle(positions)
    return [task_pool[position] for position in positions]


def create_episode(
    task_pool,
    num_shots_range,
    num_queries_per_class,
    num_ways_range=(2, 5),
    shuffle_labels=False,
):
    """Create classification support/query sets with disjoint case IDs."""
    num_ways = random.randint(num_ways_range[0], num_ways_range[1])
    num_shots = random.randint(num_shots_range[0], num_shots_range[1])
    cases = sorted({str(item[2]) for item in task_pool})
    if len(cases) < 2:
        return None

    split = None
    for _ in range(20):
        shuffled_cases = cases[:]
        random.shuffle(shuffled_cases)
        cut = min(max(1, len(shuffled_cases) // 2), len(shuffled_cases) - 1)
        support_cases = set(shuffled_cases[:cut])
        support_by_class, query_by_class = defaultdict(list), defaultdict(list)
        for seq, label, case_id in task_pool:
            destination = (
                support_by_class if str(case_id) in support_cases else query_by_class
            )
            destination[label].append((seq, label, case_id))
        available = [
            label
            for label in support_by_class.keys() & query_by_class.keys()
            if len(support_by_class[label]) >= num_shots
            and len(query_by_class[label]) >= num_queries_per_class
        ]
        if len(available) >= num_ways:
            split = support_by_class, query_by_class, available
            break
    if split is None:
        return None

    support_by_class, query_by_class, available = split
    episode_classes = random.sample(available, num_ways)
    label_map = {}
    if shuffle_labels:
        shuffled_classes = random.sample(episode_classes, len(episode_classes))
        label_map = dict(zip(episode_classes, shuffled_classes))
    support_set, query_set = [], []
    for label in episode_classes:
        mapped = label_map.get(label, label)
        support_set.extend(
            (item[0], mapped)
            for item in random.sample(support_by_class[label], num_shots)
        )
        query_set.extend(
            (item[0], mapped)
            for item in random.sample(query_by_class[label], num_queries_per_class)
        )
    random.shuffle(support_set)
    random.shuffle(query_set)
    return support_set, query_set
