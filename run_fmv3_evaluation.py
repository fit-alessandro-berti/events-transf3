#!/usr/bin/env python3
"""Evaluate the checkpoint manifest concurrently across the four GPUs."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from config_utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/fmv3/manifest.yaml")
    parser.add_argument("--checkpoint_root", default="checkpoints/fmv3")
    parser.add_argument("--output_root", default="evaluation_results/fmv3")
    parser.add_argument("--log_root", default="evaluation_output/fmv3")
    parser.add_argument("--logs_dir", default="logs_eval")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    variants = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["variants"]
    jobs = queue.Queue()
    for relative in variants:
        cfg = load_yaml_config(str((manifest_path.parent / relative).resolve()))
        jobs.put(cfg["experiment_name"])
    output_root, log_root = (root / args.output_root).resolve(), (root / args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    failures, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                name = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                sys.executable, str(root / "evaluate_fmv3.py"),
                "--checkpoint_dir", str((root / args.checkpoint_root / name).resolve()),
                "--logs_dir", str((root / args.logs_dir).resolve()),
                "--output_dir", str(output_root / name), "--device", "cuda:0",
            ]
            if args.resume:
                command.append("--resume")
            for override in args.overrides:
                command.extend(["--set", override])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_root / f"{name}.log"
            print(f"[GPU {gpu}] evaluating {name}", flush=True)
            with log_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
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
        raise SystemExit(f"Evaluation failures: {failures}")


if __name__ == "__main__":
    main()
