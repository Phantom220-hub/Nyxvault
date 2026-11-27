#!/usr/bin/env python3
"""Train NYXVAULT GPT from random initialization with resumable checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    from model import NYXVAULTGPT
except ImportError:
    from training.model import NYXVAULTGPT


def learning_rate_at_step(
    step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float
) -> float:
    """Linear warmup followed by cosine decay to base_lr * min_lr_ratio."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    step = max(0, min(step, total_steps - 1))
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coefficient)


def resolve_precision(requested: str, device: torch.device) -> Tuple[torch.dtype | None, bool]:
    """Return autocast dtype and whether CUDA GradScaler is required."""
    if requested == "fp32" or device.type != "cuda":
        return None, False
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 requested, but this CUDA GPU does not support bfloat16.")
        return torch.bfloat16, False
    if requested == "fp16":
        return torch.float16, True
    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    raise ValueError(f"Unsupported precision: {requested}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the NYXVAULT Transformer from random initialization."
    )
    parser.add_argument("--data", default="training_data")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--steps", type=int, default=6000, help="total optimizer update steps for the full run")
    parser.add_argument("--max-steps", type=int, default=0, help="maximum new optimizer steps in this invocation; 0 runs to --steps")
    parser.add_argument("--batch-size", type=int, default=8, help="micro-batch size")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10, help="print training progress every N optimizer steps")
    parser.add_argument("--save-first-checkpoint", type=int, default=100, help="save a resumable checkpoint at this early milestone")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_byte_tensor(value: Any) -> torch.Tensor:
    """Normalize serialized RNG state for PyTorch versions requiring CPU uint8 tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(_rng_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [_rng_byte_tensor(item) for item in state["torch_cuda"]]
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("--batch-size and --grad-accum-steps must be positive")
    if args.block_size < 2:
        parser.error("--block-size must be at least 2")
    if args.d_model < 1 or args.n_heads < 1 or args.n_layers < 1:
        parser.error("--d-model, --n-heads and --n-layers must be positive")
    if args.d_model % args.n_heads != 0:
        parser.error("--d-model must be divisible by --n-heads")
    if args.save_every < 1 or args.eval_interval < 1 or args.eval_batches < 1 or args.log_interval < 1:
        parser.error("--save-every, --eval-interval, --eval-batches and --log-interval must be positive")
    if args.save_first_checkpoint < 1:
        parser.error("--save-first-checkpoint must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        parser.error("--warmup-steps must be >= 0 and smaller than --steps")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("--lr and --grad-clip must be positive; --weight-decay cannot be negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be between 0 and 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'tokenizers'. Install project training requirements first."
        ) from exc

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    data_dir = Path(args.data)
    train_path, validation_path = data_dir / "train.txt", data_dir / "validation.txt"
    if not train_path.is_file() or not validation_path.is_file():
        raise SystemExit(
            f"Expected {train_path} and {validation_path}. Run training/prepare_data.py first."
        )

    def encode(path: Path) -> torch.Tensor:
        text = path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        return torch.tensor(ids, dtype=torch.long)

    train_tokens = encode(train_path)
    validation_tokens = encode(validation_path)
    if len(train_tokens) < args.block_size + 2:
        raise SystemExit("Training corpus is too small for --block-size.")
    if len(validation_tokens) < args.block_size + 2:
        raise SystemExit("Validation corpus is too small for --block-size.")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requested, but CUDA is not available")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            parser.error("--device mps requested, but MPS is not available")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    autocast_dtype, use_scaler = resolve_precision(args.precision, device)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        # PyTorch 2.1/2.2 exposes the CUDA scaler through the older namespace.
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    cfg = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "block_size": args.block_size,
        "dropout": 0.1,
    }
    model = NYXVAULTGPT(vocab_size, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_validation_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if checkpoint.get("vocab_size") != vocab_size or checkpoint.get("config") != cfg:
            raise SystemExit(
                "Checkpoint vocabulary/config does not match current tokenizer/model settings."
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise SystemExit(
                "This checkpoint has model weights only and cannot resume optimizer state. "
                "Use a step_*.pt checkpoint rather than best.pt."
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"]) + 1
        best_validation_loss = float(checkpoint.get("best_val", best_validation_loss))
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
        if "batch_rng_state" in checkpoint:
            train_generator.set_state(_rng_byte_tensor(checkpoint["batch_rng_state"]))
        if start_step >= args.steps:
            raise SystemExit(
                f"Checkpoint is already at step {start_step}; set --steps to a larger total."
            )
        print(f"Resumed from {resume_path} at optimizer step {start_step}", flush=True)

    def make_batch(tokens: torch.Tensor, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(tokens) - args.block_size - 1
        indices = torch.randint(
            0, max_start + 1, (args.batch_size,), generator=generator
        )
        x = torch.stack([tokens[i : i + args.block_size] for i in indices])
        y = torch.stack([tokens[i + 1 : i + args.block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    def autocast_context():
        if autocast_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    @torch.no_grad()
    def estimate_losses() -> Tuple[float, float]:
        model.eval()
        values = []
        for tokens in (train_tokens, validation_tokens):
            losses = []
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 1000 + len(losses) + (0 if tokens is train_tokens else 1)
            )
            for _ in range(args.eval_batches):
                x, y = make_batch(tokens, generator)
                with autocast_context():
                    _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Evaluation produced a non-finite loss.")
                losses.append(float(loss.item()))
            values.append(float(np.mean(losses)))
        model.train()
        return values[0], values[1]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    end_step = args.steps if args.max_steps == 0 else min(args.steps, start_step + args.max_steps)
    if end_step <= start_step:
        raise SystemExit(f"No training steps left to run: start={start_step}, end={end_step}.")
    print(f"Training range: {start_step + 1}..{end_step} of {args.steps}", flush=True)
    for step in range(start_step, end_step):
        learning_rate = learning_rate_at_step(
            step, args.steps, args.lr, args.warmup_steps, args.min_lr_ratio
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _micro_step in range(args.grad_accum_steps):
            x, y = make_batch(train_tokens, train_generator)
            with autocast_context():
                _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Training produced a non-finite loss at step {step}.")
            scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().item())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        scaler.step(optimizer)
        scaler.update()

        # Report that an optimizer update completed before running the slower evaluation pass.
        if (step + 1) % args.log_interval == 0 or step == start_step or step + 1 == args.steps:
            print(
                f"training step {step + 1}/{args.steps} "
                f"micro_loss={accumulated_loss / args.grad_accum_steps:.4f} "
                f"grad_norm={float(grad_norm):.4f}",
                flush=True,
            )

        if (step + 1) % args.eval_interval == 0 or step == start_step:
            train_loss, validation_loss = estimate_losses()
            validation_ppl = math.exp(min(validation_loss, 20.0))
            print(
                f"step {step + 1}/{args.steps} "
                f"lr={learning_rate:.7g} "
                f"micro_loss={accumulated_loss / args.grad_accum_steps:.4f} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"val_ppl={validation_ppl:.2f}",
                flush=True,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "vocab_size": vocab_size,
                        "step": step,
                        "best_val": best_validation_loss,
                        "training": {
                            "seed": args.seed,
                            "precision": args.precision,
                            "grad_accum_steps": args.grad_accum_steps,
                            "lr": args.lr,
                            "warmup_steps": args.warmup_steps,
                            "min_lr_ratio": args.min_lr_ratio,
                        },
                    },
                    output_dir / "best.pt",
                )

        if (step + 1) == args.save_first_checkpoint or (step + 1) % args.save_every == 0 or step + 1 == end_step:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "vocab_size": vocab_size,
                "step": step,
                "best_val": best_validation_loss,
                "rng_state": _rng_state(),
                "batch_rng_state": train_generator.get_state(),
                "training": {
                    "seed": args.seed,
                    "precision": args.precision,
                    "grad_accum_steps": args.grad_accum_steps,
                    "lr": args.lr,
                    "warmup_steps": args.warmup_steps,
                    "min_lr_ratio": args.min_lr_ratio,
                    "total_steps": args.steps,
                },
            }
            checkpoint_path = output_dir / f"step_{step + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)

    if end_step < args.steps:
        print(f"Training paused safely at step {end_step}/{args.steps}. Resume from step_{end_step}.pt.", flush=True)
    else:
        print(f"Training finished. Best validation loss: {best_validation_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
