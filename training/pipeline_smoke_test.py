#!/usr/bin/env python3
"""Run the real NYXVAULT trainer briefly on prepared corpus data using CPU.

This is an integration test, not a quality benchmark. It exercises train.py,
validation loss, checkpoint writing, and optimizer-state resume.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_training(command: list[str], label: str) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/data/processed")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()

    if args.steps < 2 or args.steps > 20:
        parser.error("--steps must be between 2 and 20")
    if args.block_size < 2:
        parser.error("--block-size must be at least 2")

    data_dir = repo_path(args.data)
    tokenizer = repo_path(args.tokenizer)
    train_path = data_dir / "train.txt"
    validation_path = data_dir / "validation.txt"
    for required in (train_path, validation_path, tokenizer):
        if not required.is_file():
            raise SystemExit(f"Required pipeline input is missing: {required}")

    if train_path.stat().st_size < 100 or validation_path.stat().st_size < 40:
        raise SystemExit("Prepared train/validation corpus is unexpectedly small.")

    with tempfile.TemporaryDirectory(prefix="nyxvault-pipeline-smoke-") as temp:
        output_dir = Path(temp) / "checkpoints"
        train_script = str(ROOT / "training" / "train.py")
        common = [
            sys.executable, train_script,
            "--data", str(data_dir),
            "--tokenizer", str(tokenizer),
            "--out", str(output_dir),
            "--batch-size", "2",
            "--grad-accum-steps", "1",
            "--block-size", str(args.block_size),
            "--d-model", "32",
            "--n-heads", "4",
            "--n-layers", "2",
            "--lr", "0.0005",
            "--warmup-steps", "1",
            "--min-lr-ratio", "0.1",
            "--precision", "fp32",
            "--eval-interval", "1",
            "--eval-batches", "1",
            "--save-every", str(args.steps),
            "--seed", "2026",
        ]

        print("Stage 1/2: CPU training on the prepared corpus")
        run_training(common + ["--steps", str(args.steps)], "initial training")
        first_checkpoint = output_dir / f"step_{args.steps}.pt"
        if not first_checkpoint.is_file():
            raise RuntimeError(f"Trainer did not create checkpoint: {first_checkpoint}")

        print("Stage 2/2: resume the same run from its optimizer checkpoint")
        resumed_steps = args.steps + 1
        resume_command = [
            item for item in common
            if item not in ("--save-every", str(args.steps))
        ]
        resume_command += [
            "--save-every", str(resumed_steps),
            "--steps", str(resumed_steps),
            "--resume", str(first_checkpoint),
        ]
        resume_log = run_training(resume_command, "checkpoint resume")
        if "Resumed from" not in resume_log:
            raise RuntimeError("Trainer did not report restoring the checkpoint.")
        final_checkpoint = output_dir / f"step_{resumed_steps}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"Resumed trainer did not create: {final_checkpoint}")

    print("NYXVAULT real-data CPU pipeline integration: PASS")
    print("Prepared corpus + tokenizer + train.py + validation + checkpoint + resume.")
    print("This confirms pipeline mechanics, not coding ability or answer quality.")


if __name__ == "__main__":
    main()
