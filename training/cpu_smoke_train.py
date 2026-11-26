#!/usr/bin/env python3
"""Tiny CPU-only training smoke test. Does not use real training data or produce usable weights."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

try:
    from model import NYXVAULTGPT
except ImportError:
    from training.model import NYXVAULTGPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg = {"d_model": 32, "n_heads": 4, "n_layers": 2, "block_size": 16, "dropout": 0.0}
    model = NYXVAULTGPT(vocab_size=32, cfg=cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Repeated toy token sequence: this is only a mechanics test, not a coding dataset.
    sequence = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 4)
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    model.eval()
    with torch.no_grad():
        _, initial_loss = model(inputs, targets)
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, final_loss = model(inputs, targets)

    if not final_loss.item() < initial_loss.item():
        raise RuntimeError(f"Loss did not decrease: {initial_loss.item()} -> {final_loss.item()}")

    save_path = Path(args.checkpoint) if args.checkpoint else Path(tempfile.mkdtemp()) / "smoke_checkpoint.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.steps, "config": cfg, "vocab_size": 32}, save_path)
    restored = torch.load(save_path, map_location="cpu", weights_only=False)
    resumed = NYXVAULTGPT(vocab_size=32, cfg=cfg).cpu()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-3)
    resumed.load_state_dict(restored["model"])
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed.train()
    resumed_optimizer.zero_grad(set_to_none=True)
    _, resumed_loss = resumed(inputs, targets)
    resumed_loss.backward()
    resumed_optimizer.step()

    print(f"CPU smoke test PASS: {args.steps} steps; loss {initial_loss.item():.4f} -> {final_loss.item():.4f}")
    print(f"Checkpoint saved and optimizer state resumed successfully: {save_path}")
    print("NOTE: toy data only; this does not demonstrate programming knowledge or model quality.")


if __name__ == "__main__":
    main()
