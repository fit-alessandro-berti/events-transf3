"""Per-log sequence and linear baselines for the FM-v3 case-budget protocol."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd
import pm4py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, Dataset


def load_classification_tasks(path, max_seq_len=10):
    df = pm4py.read_xes(path)
    activity_key, time_key, case_key = "concept:name", "time:timestamp", "case:concept:name"
    resource_key, cost_key = "org:resource", "amount"
    df[time_key] = pd.to_datetime(df[time_key], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[time_key])
    activities = sorted(str(value) for value in df[activity_key].dropna().unique())
    activity_to_id = {name: idx for idx, name in enumerate(activities)}
    tasks = []
    for case_id, trace in df.groupby(case_key):
        trace = trace.sort_values(time_key)
        if len(trace) < 3:
            continue
        start, previous, events = trace.iloc[0][time_key], trace.iloc[0][time_key], []
        for _, event in trace.iterrows():
            timestamp = event[time_key]
            cost = event.get(cost_key, 0.0)
            cost = float(cost) if pd.notna(cost) and isinstance(cost, (int, float, np.number)) else 0.0
            events.append({
                "activity_id": activity_to_id[str(event[activity_key])],
                "cost": cost,
                "time_from_start": (timestamp - start).total_seconds(),
                "time_from_previous": (timestamp - previous).total_seconds(),
            })
            previous = timestamp
        for index in range(1, len(events) - 1):
            tasks.append((events[max(0, index + 1 - max_seq_len) : index + 1], events[index + 1]["activity_id"], str(case_id)))
    return tasks, activities


class PrefixDataset(Dataset):
    def __init__(self, tasks, indices):
        self.tasks = tasks
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        sequence, label, _ = self.tasks[self.indices[position]]
        activities = torch.as_tensor([event["activity_id"] + 1 for event in sequence], dtype=torch.long)
        numerical = torch.as_tensor([
            [np.log1p(max(0.0, float(event["cost"]))),
             np.log1p(max(0.0, float(event["time_from_start"]))),
             np.log1p(max(0.0, float(event["time_from_previous"])))]
            for event in sequence
        ], dtype=torch.float32)
        return activities, numerical, int(label)


def collate_prefixes(batch):
    lengths = torch.as_tensor([len(item[0]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    activities = torch.zeros((len(batch), max_length), dtype=torch.long)
    numerical = torch.zeros((len(batch), max_length, 3), dtype=torch.float32)
    labels = torch.as_tensor([item[2] for item in batch], dtype=torch.long)
    for row, (activity, numeric, _) in enumerate(batch):
        activities[row, : len(activity)] = activity
        numerical[row, : len(activity)] = numeric
    return activities, numerical, lengths, labels


class NextActivityLSTM(nn.Module):
    def __init__(self, num_classes, embedding_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.activity_embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)
        self.numeric_norm = nn.LayerNorm(3)
        self.lstm = nn.LSTM(embedding_dim + 3, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, activities, numerical, lengths):
        inputs = torch.cat([self.activity_embedding(activities), self.numeric_norm(numerical)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1]))


def train_lstm(tasks, support_indices, num_classes, loss_mode, config, device, seed):
    torch.manual_seed(seed)
    model = NextActivityLSTM(
        num_classes,
        int(config.get("embedding_dim", 32)),
        int(config.get("hidden_dim", 64)),
        float(config.get("dropout", 0.1)),
    ).to(device)
    labels = np.asarray([tasks[int(index)][1] for index in support_indices], dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    nonzero = counts > 0
    class_weights = np.zeros(num_classes, dtype=np.float32)
    class_weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    weights = torch.as_tensor(class_weights, device=device)
    log_counts = torch.log(torch.as_tensor(counts + 1.0, dtype=torch.float32, device=device))
    loader = DataLoader(
        PrefixDataset(tasks, support_indices), batch_size=int(config.get("batch_size", 64)),
        shuffle=True, collate_fn=collate_prefixes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=1e-4)
    model.train()
    for _ in range(int(config.get("epochs", 20))):
        for activities, numerical, lengths, targets in loader:
            activities, numerical, lengths, targets = (
                activities.to(device), numerical.to(device), lengths.to(device), targets.to(device)
            )
            logits = model(activities, numerical, lengths)
            if loss_mode == "class_weighted_ce":
                loss = F.cross_entropy(logits, targets, weight=weights)
            elif loss_mode == "balanced_softmax":
                loss = F.cross_entropy(logits + log_counts.unsqueeze(0), targets)
            else:
                loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, counts


@torch.no_grad()
def predict_lstm(model, tasks, query_indices, counts, mode, config, device):
    model.eval()
    loader = DataLoader(
        PrefixDataset(tasks, query_indices), batch_size=int(config.get("batch_size", 64)),
        shuffle=False, collate_fn=collate_prefixes,
    )
    outputs = []
    log_prior = torch.log(torch.as_tensor(counts + float(config.get("prior_smoothing", 1.0)), device=device))
    for activities, numerical, lengths, _ in loader:
        logits = model(activities.to(device), numerical.to(device), lengths.to(device))
        if mode == "logit_adjustment":
            logits = logits - float(config.get("logit_adjustment_tau", 1.0)) * log_prior.unsqueeze(0)
        outputs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def handcrafted_features(tasks, indices, num_classes):
    rows = []
    for index in indices:
        sequence = tasks[int(index)][0]
        counts = np.bincount([event["activity_id"] for event in sequence], minlength=num_classes).astype(float)
        counts /= max(len(sequence), 1)
        last = np.zeros(num_classes)
        last[sequence[-1]["activity_id"]] = 1.0
        temporal = [
            np.log1p(max(0.0, float(sequence[-1]["time_from_start"]))),
            np.log1p(max(0.0, float(sequence[-1]["time_from_previous"]))),
            np.log1p(max(0.0, float(sequence[-1]["cost"]))),
            len(sequence) / 10.0,
        ]
        rows.append(np.concatenate([counts, last, temporal]))
    return np.asarray(rows)


def predict_weighted_linear(tasks, support_indices, query_indices, num_classes, seed):
    y = np.asarray([tasks[int(index)][1] for index in support_indices], dtype=int)
    if len(np.unique(y)) == 1:
        probabilities = np.zeros((len(query_indices), num_classes))
        probabilities[:, int(y[0])] = 1.0
        return probabilities
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    model.fit(handcrafted_features(tasks, support_indices, num_classes), y)
    partial = model.predict_proba(handcrafted_features(tasks, query_indices, num_classes))
    probabilities = np.zeros((len(query_indices), num_classes))
    probabilities[:, model.classes_.astype(int)] = partial
    return probabilities


def predict_classical(tasks, support_indices, query_indices, num_classes, seed, mode):
    """Small-data classical comparators on a fixed, non-foundation representation."""
    y = np.asarray([tasks[int(index)][1] for index in support_indices], dtype=int)
    if len(np.unique(y)) == 1:
        probabilities = np.zeros((len(query_indices), num_classes))
        probabilities[:, int(y[0])] = 1.0
        return probabilities
    if mode == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=1,
            random_state=seed, n_jobs=1,
        )
    elif mode == "gaussian_nb":
        model = GaussianNB(var_smoothing=1e-8)
    else:
        raise ValueError(f"Unknown classical baseline mode: {mode}")
    model.fit(handcrafted_features(tasks, support_indices, num_classes), y)
    partial = model.predict_proba(handcrafted_features(tasks, query_indices, num_classes))
    probabilities = np.zeros((len(query_indices), num_classes))
    probabilities[:, model.classes_.astype(int)] = partial
    return probabilities


def predict_tabpfn(tasks, support_indices, query_indices, num_classes, config, device, seed):
    """TabPFN on the same handcrafted prefix representation used by linear baselines."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    y = np.asarray([tasks[int(index)][1] for index in support_indices], dtype=int)
    if len(np.unique(y)) == 1:
        probabilities = np.zeros((len(query_indices), num_classes))
        probabilities[:, int(y[0])] = 1.0
        return probabilities
    version_name = str(config.get("model_version", "v2")).upper().replace(".", "_")
    version = getattr(ModelVersion, version_name)
    model = TabPFNClassifier.create_default_for_version(
        version,
        n_estimators=int(config.get("n_estimators", 4)),
        device=str(device),
        ignore_pretraining_limits=True,
        balance_probabilities=bool(config.get("balance_probabilities", False)),
        random_state=seed,
        show_progress_bar=False,
    )
    if len(np.unique(y)) > int(config.get("max_native_classes", 10)):
        from tabpfn_extensions.many_class import ManyClassClassifier

        model = ManyClassClassifier(
            estimator=model,
            alphabet_size=int(config.get("max_native_classes", 10)),
            n_estimators_redundancy=int(config.get("many_class_redundancy", 2)),
            random_state=seed,
            verbose=0,
        )
    model.fit(handcrafted_features(tasks, support_indices, num_classes), y)
    partial = model.predict_proba(handcrafted_features(tasks, query_indices, num_classes))
    probabilities = np.zeros((len(query_indices), num_classes))
    probabilities[:, model.classes_.astype(int)] = partial
    return probabilities
