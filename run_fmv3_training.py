#!/usr/bin/env python3
"""Run the complete FM-v3 checkpoint manifest, one process per GPU."""

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
    parser.add_argument("--log_root", default="training_output/fmv3")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_after_epoch", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    jobs = queue.Queue()
    for relative in manifest["variants"]:
        config_path = (manifest_path.parent / relative).resolve()
        config = load_yaml_config(str(config_path))
        jobs.put((config["experiment_name"], config_path))

    checkpoint_root = (root / args.checkpoint_root).resolve()
    log_root = (root / args.log_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                name, config_path = jobs.get_nowait()
            except queue.Empty:
                return
            checkpoint_dir = checkpoint_root / name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(root / "main.py"), "--config", str(config_path),
                "--checkpoint_dir", str(checkpoint_dir), "--cleanup_checkpoints",
            ]
            if args.resume:
                command.append("--resume")
            if args.stop_after_epoch is not None:
                command.extend(["--stop_after_epoch", str(args.stop_after_epoch)])
            for override in args.overrides:
                command.extend(["--set", override])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_root / f"{name}.log"
            print(f"[GPU {gpu}] starting {name} -> {checkpoint_dir}", flush=True)
            with log_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
                handle.write("COMMAND: " + " ".join(command) + "\n")
                handle.flush()
                result = subprocess.run(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                with lock:
                    failures.append((name, result.returncode))
                print(f"[GPU {gpu}] FAILED {name} ({result.returncode})", flush=True)
            else:
                print(f"[GPU {gpu}] complete {name}", flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"Training failures: {failures}")
    print("All manifest checkpoints completed.")


if __name__ == "__main__":
    main()
