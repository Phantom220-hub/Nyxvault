#!/usr/bin/env python3
"""Supervised instruction fine-tuning (SFT) for NYXVAULT."""
from __future__ import annotations
import argparse, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch

try:
    from model import NYXVAULTGPT
    from data_schema import normalize_record, validate_record, render_prompt
except ImportError:
    from training.model import NYXVAULTGPT
    from training.data_schema import normalize_record, validate_record, render_prompt


def byte_state(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def make_samples(path: Path, tokenizer, block_size: int, seed: int):
    eos_id, pad_id = tokenizer.token_to_id("[EOS]"), tokenizer.token_to_id("[PAD]")
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must contain [EOS] and [PAD].")
    buckets = defaultdict(lambda: {"train": [], "valid": []})
    seen_ids, seen_tasks = set(), set()
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rec = normalize_record(json.loads(line))
                issues = [x for x in validate_record(rec) if x.severity == "error" or x.code == "W010"]
                if issues:
                    raise ValueError("; ".join(x.message for x in issues[:3]))
                if rec["id"] in seen_ids:
                    raise ValueError("duplicate id " + rec["id"])
                task_key = (rec["instruction"], rec.get("input", ""))
                if task_key in seen_tasks:
                    raise ValueError("duplicate task " + rec["id"])
                seen_ids.add(rec["id"])
                seen_tasks.add(task_key)
                prompt_ids = tokenizer.encode(render_prompt(rec)).ids
                answer_ids = tokenizer.encode(rec["output"]).ids + [eos_id]
                if len(answer_ids) > block_size - 8:
                    raise ValueError(f"answer exceeds block_size={block_size}")
                prompt_ids = prompt_ids[-(block_size + 1 - len(answer_ids)):]
                if not prompt_ids:
                    raise ValueError("prompt tokenization is empty")
                joined = prompt_ids + answer_ids
                input_ids, labels = joined[:-1], joined[1:]
                labels[:max(0, len(prompt_ids) - 1)] = [-100] * max(0, len(prompt_ids) - 1)
                input_ids += [pad_id] * (block_size - len(input_ids))
                labels += [-100] * (block_size - len(labels))
                sample = (input_ids, labels)
                category = (rec.get("tags") or ["uncategorized"])[0]
                # Hash-based stratification prevents a sorted dataset from putting
                # only its first template into validation.
                bucket = int(hashlib.sha256(rec["id"].encode()).hexdigest()[:8], 16) % 10
                buckets[category]["valid" if bucket == 0 else "train"].append(sample)
            except Exception as exc:
                errors.append(f"line {line_no}: {exc}")
    if errors:
        raise ValueError("SFT dataset validation failed:\n" + "\n".join(errors[:25]))
    train = [sample for group in buckets.values() for sample in group["train"]]
    valid = [sample for group in buckets.values() for sample in group["valid"]]
    if not train or not valid:
        raise ValueError("SFT train/validation split is empty.")
    counts = {key: {"train": len(value["train"]), "validation": len(value["valid"])}
              for key, value in sorted(buckets.items())}
    return train, valid, counts, pad_id


def batch(samples, indices, device):
    x = torch.tensor([samples[i][0] for i in indices], dtype=torch.long, device=device)
    y = torch.tensor([samples[i][1] for i in indices], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(model, samples, device, batch_size, limit=200):
    model.eval()
    chosen = samples
    if len(chosen) > limit:
        ids = sorted(random.Random(20261127).sample(range(len(chosen)), limit))
        chosen = [chosen[i] for i in ids]
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(chosen), batch_size):
        idx = list(range(start, min(start + batch_size, len(chosen))))
        x, y = batch(chosen, idx, device)
        _, loss = model(x, y)
        count = int((y != -100).sum().item())
        total_loss += float(loss.item()) * count
        total_tokens += count
    model.train()
    return total_loss / max(total_tokens, 1)


def save_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description="NYXVAULT instruction supervised fine-tuning")
    p.add_argument("--data", default="training/data/instruction_sft.jsonl")
    p.add_argument("--tokenizer", default="/content/nyxvault-output/tokenizer.json")
    p.add_argument("--init-from", default="", help="base checkpoint for the first SFT stage")
    p.add_argument("--resume", default="", help="full SFT checkpoint for a later stage")
    p.add_argument("--out", default="/content/nyxvault-output/sft_checkpoints")
    p.add_argument("--steps", type=int, default=1000, help="total optimizer steps in the whole SFT plan")
    p.add_argument("--max-steps", type=int, default=100, help="new steps in this invocation")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--allow-cpu", action="store_true", help="allow slow CPU training when CUDA is unavailable")
    args = p.parse_args()
    if args.steps < 1 or args.max_steps < 1 or args.max_steps > args.steps:
        p.error("steps/max-steps must be positive and max-steps <= steps")
    if args.batch_size < 1 or args.grad_accum_steps < 1 or args.block_size < 16:
        p.error("batch-size/grad-accum-steps must be positive and block-size >= 16")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        p.error("warmup-steps must be in [0, steps)")
    if args.lr <= 0 or args.eval_every < 1 or args.save_every < 1:
        p.error("lr, eval-every and save-every must be positive")
    if bool(args.init_from) == bool(args.resume):
        p.error("provide exactly one of --init-from or --resume")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU not found. Enable Colab GPU or pass --allow-cpu intentionally.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    train_samples, valid_samples, counts, _pad_id = make_samples(
        Path(args.data), tokenizer, args.block_size, args.seed
    )
    print("Examples by category:", json.dumps(counts, ensure_ascii=False), flush=True)
    print(f"Total examples: {len(train_samples) + len(valid_samples)}; train={len(train_samples)}; validation={len(valid_samples)}", flush=True)

    initial_path = args.resume or args.init_from
    loaded = torch.load(initial_path, map_location="cpu", weights_only=False)
    cfg, vocab_size = loaded["config"], int(loaded["vocab_size"])
    if vocab_size != tokenizer.get_vocab_size():
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch.")
    if cfg["block_size"] != args.block_size:
        raise ValueError("block-size must match the base checkpoint.")
    model = NYXVAULTGPT(vocab_size, cfg).to(device)
    model.load_state_dict(loaded["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else None)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    sampler = random.Random(args.seed)
    start_step, best_loss = 0, float("inf")

    if args.resume:
        if "optimizer" not in loaded:
            raise ValueError("SFT resume file has no optimizer state; use sft_step_N.pt.")
        optimizer.load_state_dict(loaded["optimizer"])
        if "scaler" in loaded: scaler.load_state_dict(loaded["scaler"])
        start_step = int(loaded["step"]) + 1
        best_loss = float(loaded.get("best_validation_loss", best_loss))
        if "sample_rng_state" in loaded: sampler.setstate(loaded["sample_rng_state"])
        if "python_rng" in loaded: random.setstate(loaded["python_rng"])
        if "numpy_rng" in loaded: np.random.set_state(loaded["numpy_rng"])
        if "torch_rng" in loaded: torch.set_rng_state(byte_state(loaded["torch_rng"]))
        if device.type == "cuda" and "cuda_rng" in loaded:
            torch.cuda.set_rng_state_all([byte_state(x) for x in loaded["cuda_rng"]])
        print(f"Resumed SFT from {initial_path} at step {start_step}", flush=True)
    else:
        print(f"Starting SFT from base checkpoint: {initial_path}", flush=True)

    if start_step >= args.steps:
        raise ValueError(f"SFT already reached step {start_step}; increase --steps.")
    end_step = min(args.steps, start_step + args.max_steps)
    print(f"SFT training range: {start_step + 1}..{end_step} of {args.steps}", flush=True)

    def autocast_ctx():
        if amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    model.train()
    for step in range(start_step, end_step):
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
        if step >= args.warmup_steps:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            lr *= 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            indices = [sampler.randrange(len(train_samples)) for _ in range(args.batch_size)]
            x, y = batch(train_samples, indices, device)
            with autocast_ctx(): _, loss = model(x, y)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite SFT loss at step {step + 1}")
            scaler.scale(loss / args.grad_accum_steps).backward()
            accum_loss += float(loss.detach().item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"non-finite gradient norm at step {step + 1}")
        scaler.step(optimizer); scaler.update()

        if (step + 1) % 10 == 0 or step == start_step:
            print(f"sft step {step + 1}/{args.steps} train_loss={accum_loss / args.grad_accum_steps:.4f} grad_norm={float(grad_norm):.4f}", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == end_step:
            val_loss = evaluate(model, valid_samples, device, args.batch_size)
            print(f"sft validation step={step + 1} answer_loss={val_loss:.5f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                save_atomic(Path(args.out) / "best_sft.pt", {
                    "model": model.state_dict(), "config": cfg, "vocab_size": vocab_size,
                    "step": step, "best_validation_loss": best_loss, "phase": "instruction_sft",
                    "base_checkpoint": args.init_from or loaded.get("base_checkpoint", ""),
                    "tokenizer_vocab_size": tokenizer.get_vocab_size(),
                })

        if (step + 1) % args.save_every == 0 or step + 1 == end_step:
            payload = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg, "vocab_size": vocab_size, "step": step,
                "best_validation_loss": best_loss, "phase": "instruction_sft",
                "sample_rng_state": sampler.getstate(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "training": {"total_steps": args.steps, "lr": args.lr, "warmup_steps": args.warmup_steps,
                             "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
                             "block_size": args.block_size},
            }
            if device.type == "cuda": payload["cuda_rng"] = torch.cuda.get_rng_state_all()
            path = Path(args.out) / f"sft_step_{step + 1}.pt"
            save_atomic(path, payload)
            print(f"SFT checkpoint saved: {path}", flush=True)

    if end_step < args.steps:
        print(f"SFT paused safely at step {end_step}/{args.steps}.", flush=True)
    else:
        print(f"SFT finished. Best validation answer loss: {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
