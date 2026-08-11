#!/usr/bin/env python3
"""Evaluate low-data baselines concurrently, assigning whole logs to GPUs."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def _log_stem(path: Path) -> str:
    return path.name.replace(".xes.gz", "").replace(".xes", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fmv3/baselines.yaml")
    parser.add_argument("--logs_dir", default="logs_eval")
    parser.add_argument("--output_root", default="evaluation_results/fmv3/baselines")
    parser.add_argument("--log_root", default="evaluation_output/fmv3/baselines")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    logs_dir = (root / args.logs_dir).resolve()
    paths = sorted(list(logs_dir.glob("*.xes")) + list(logs_dir.glob("*.xes.gz")))
    if not paths:
        raise FileNotFoundError(f"No XES logs found in {logs_dir}")

    jobs = queue.Queue()
    for path in paths:
        jobs.put(path)
    output_root = (root / args.output_root).resolve()
    log_root = (root / args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str):
        while True:
            try:
                path = jobs.get_nowait()
            except queue.Empty:
                return
            name = _log_stem(path)
            command = [
                sys.executable,
                str(root / "evaluate_low_data_baselines.py"),
                "--config",
                str((root / args.config).resolve()),
                "--logs_dir",
                str(logs_dir),
                "--logs",
                name,
                "--output_dir",
                str(output_root / name),
                "--device",
                "cuda:0",
            ]
            if args.resume:
                command.append("--resume")
            for override in args.overrides:
                command.extend(["--set", override])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_root / f"{name}.log"
            print(f"[GPU {gpu}] evaluating baselines for {name}", flush=True)
            with log_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
                handle.write("COMMAND: " + " ".join(command) + "\n")
                handle.flush()
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                with lock:
                    failures.append((name, result.returncode))
                print(f"[GPU {gpu}] FAILED {name} ({result.returncode})", flush=True)
            else:
                print(f"[GPU {gpu}] complete {name}", flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"Baseline evaluation failures: {failures}")
    print("All per-log baseline evaluations completed.")


if __name__ == "__main__":
    main()
