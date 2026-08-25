#!/usr/bin/env python3
"""Generate multi-scale, attribute-rich process logs for foundation training."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ACTIVITIES = {
    "start": "Register customer request",
    "validate": "Validate submitted information",
    "risk": "Assess operational risk",
    "documents": "Collect supporting documents",
    "approve": "Approve service request",
    "reject": "Reject service request",
    "review": "Review exceptional conditions",
    "fix": "Correct incomplete information",
    "escalate": "Escalate overdue request",
    "schedule": "Schedule fulfillment work",
    "fulfill": "Fulfill approved request",
    "audit": "Audit completed fulfillment",
    "refund": "Issue customer refund",
    "handover": "Transfer case ownership",
    "end": "Close customer request",
}
ROLES = ("analyst", "senior_analyst", "approver", "operator", "auditor", "vendor")
CHANNELS = ("portal", "email", "phone", "partner_api")
SCALE_HOURS = (1.0 / 60.0, 1.0, 24.0, 24.0 * 30.0)


@dataclass(frozen=True)
class SyntheticEvent:
    activity: str
    timestamp: datetime
    resource: str
    role: str
    amount: float
    lifecycle: str = "complete"


def _next_business_time(value):
    value = value.astimezone(timezone.utc)
    while value.weekday() >= 5:
        value = (value + timedelta(days=1)).replace(hour=9, minute=0, second=0)
    if value.hour < 9:
        return value.replace(hour=9, minute=0, second=0)
    if value.hour >= 17:
        return _next_business_time(
            (value + timedelta(days=1)).replace(hour=9, minute=0, second=0)
        )
    return value


def _duration(rng, scale_hours, complexity=1.0):
    heavy_tail = float(rng.lognormal(mean=-0.35, sigma=1.0))
    return timedelta(hours=max(scale_hours * complexity * heavy_tail, 1.0 / 3600.0))


def generate_trace(case_index, rng):
    """Return one realistic trace and case-level attributes."""
    scale_hours = float(rng.choice(SCALE_HOURS))
    era = int(case_index // 250)
    priority = str(rng.choice(("normal", "normal", "high", "urgent")))
    start = datetime(2022, 1, 3, 8, tzinfo=timezone.utc) + timedelta(
        hours=float(rng.uniform(0, 24 * 365 * 3))
    )
    cursor = _next_business_time(start)
    resources = {
        role: f"{role}:R{int(rng.integers(1, 31)):02d}" for role in ROLES
    }
    events = []
    cumulative_cost = 0.0

    def append(activity_key, role="analyst", complexity=1.0, amount_sign=1.0):
        nonlocal cursor, cumulative_cost
        if events and rng.random() < 0.12:
            handover_cost = float(rng.uniform(5, 50))
            cursor = _next_business_time(cursor + _duration(rng, scale_hours, 0.1))
            events.append(
                SyntheticEvent(
                    ACTIVITIES["handover"],
                    cursor,
                    resources["analyst"],
                    "analyst",
                    handover_cost,
                )
            )
            cumulative_cost += handover_cost
        cursor = _next_business_time(
            cursor + _duration(rng, scale_hours, complexity / (1.0 + 0.08 * era))
        )
        overtime = 1.5 if cursor.hour >= 17 else 1.0
        amount = amount_sign * float(
            rng.uniform(10, 150) * complexity * overtime
        )
        resource = "Unknown" if rng.random() < 0.03 else resources[role]
        events.append(
            SyntheticEvent(
                ACTIVITIES[activity_key],
                cursor,
                resource,
                role,
                amount,
            )
        )
        cumulative_cost += amount

    append("start", complexity=0.1)
    append("validate", complexity=0.7)

    # An old decision changes distant approval/fulfillment behavior.
    high_risk = rng.random() < (0.20 + 0.15 * (priority == "urgent"))
    append("risk", role="senior_analyst" if high_risk else "analyst", complexity=1.2)

    # Parallel children may share a timestamp; stable source order remains the
    # deterministic tie-breaker and the join waits for both branches.
    branch_start = cursor
    document_end = _next_business_time(branch_start + _duration(rng, scale_hours, 1.0))
    review_end = _next_business_time(branch_start + _duration(rng, scale_hours, 1.4))
    parallel = [
        SyntheticEvent(
            ACTIVITIES["documents"],
            document_end,
            resources["analyst"],
            "analyst",
            float(rng.uniform(20, 120)),
        ),
        SyntheticEvent(
            ACTIVITIES["review"],
            review_end,
            resources["senior_analyst"],
            "senior_analyst",
            float(rng.uniform(30, 180)),
        ),
    ]
    events.extend(sorted(parallel, key=lambda event: event.timestamp))
    cursor = max(document_end, review_end)
    cumulative_cost += sum(event.amount for event in parallel)

    rework_probability = 0.15 + 0.25 * high_risk
    for iteration in range(int(rng.geometric(1.0 - rework_probability)) - 1):
        append("fix", complexity=1.0 + 0.25 * iteration)
        append("review", role="senior_analyst", complexity=1.1)

    overdue = (cursor - start).total_seconds() / 3600.0 > 5.0 * scale_hours
    if overdue or cumulative_cost > 700:
        append("escalate", role="senior_analyst", complexity=0.4)

    if high_risk and rng.random() < 0.45:
        append("reject", role="approver", complexity=0.4)
        if rng.random() < 0.25:
            append("refund", role="operator", complexity=0.2, amount_sign=-1.0)
    else:
        append("approve", role="approver", complexity=0.6)
        append("schedule", role="operator", complexity=0.8)
        append("fulfill", role="vendor" if high_risk else "operator", complexity=2.0)
        # Separation of duty is encoded by the auditor resource pool.
        append("audit", role="auditor", complexity=0.8)
    append("end", complexity=0.1)

    events.sort(key=lambda event: event.timestamp)
    attributes = {
        "concept:name": f"synthetic-{case_index:07d}",
        "priority": priority,
        "channel": str(rng.choice(CHANNELS)),
        "duration_scale": ("minute", "hour", "day", "month")[
            SCALE_HOURS.index(scale_hours)
        ],
        "era": era,
        "budget": float(rng.uniform(300, 2000)),
    }
    return events, attributes


def generate_event_log(num_cases=250, seed=42):
    from pm4py.objects.log.obj import Event, EventLog, Trace

    rng = np.random.default_rng(int(seed))
    log = EventLog()
    for case_index in range(int(num_cases)):
        events, attributes = generate_trace(case_index, rng)
        trace = Trace(attributes=attributes)
        for item in events:
            trace.append(
                Event(
                    {
                        "concept:name": item.activity,
                        "time:timestamp": item.timestamp,
                        "org:resource": item.resource,
                        "org:role": item.role,
                        "lifecycle:transition": item.lifecycle,
                        "amount": float(item.amount),
                    }
                )
            )
        log.append(trace)
    return log


def validate_event_log(log):
    durations, lengths, costs = [], [], []
    for trace in log:
        if len(trace) < 2:
            raise ValueError("Every synthetic trace must contain two or more events")
        timestamps = [event["time:timestamp"] for event in trace]
        if timestamps != sorted(timestamps):
            raise ValueError("Synthetic trace timestamps are not monotone")
        for event in trace:
            required = {"concept:name", "time:timestamp", "org:resource", "amount"}
            if not required.issubset(event):
                raise ValueError(f"Synthetic event is missing {required - set(event)}")
            if not isinstance(event["amount"], float):
                raise TypeError("Synthetic event amount must be a float")
        durations.append((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0)
        lengths.append(len(trace))
        costs.extend(float(event["amount"]) for event in trace)
    return {
        "cases": len(log),
        "trace_length_quantiles": np.quantile(lengths, [0.1, 0.5, 0.9]).tolist(),
        "duration_hours_quantiles": np.quantile(
            durations, [0.1, 0.5, 0.9]
        ).tolist(),
        "negative_amounts": int(sum(value < 0 for value in costs)),
        "resources_present": int(
            sum(event["org:resource"] != "Unknown" for trace in log for event in trace)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import pm4py

    log = generate_event_log(args.cases, args.seed)
    summary = validate_event_log(log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(log, str(args.output))
    print(summary)


if __name__ == "__main__":
    main()
