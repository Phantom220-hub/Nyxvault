#!/usr/bin/env python3
"""Fast correctness checks for the NYXVAULT transformer before expensive training."""
import torch
from model import NYXVAULTGPT

def main():
    torch.manual_seed(7)
    cfg = {"d_model": 32, "n_heads": 4, "n_layers": 2, "block_size": 16, "dropout": 0.0}
    model = NYXVAULTGPT(vocab_size=64, cfg=cfg)
    model.eval()

    # Future tokens must not change logits at earlier positions.
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    changed_future = first.clone()
    changed_future[0, 4:] = torch.tensor([21, 22, 23, 24])
    with torch.no_grad():
        logits_a, _ = model(first)
        logits_b, _ = model(changed_future)
    torch.testing.assert_close(logits_a[:, :4, :], logits_b[:, :4, :], rtol=1e-5, atol=1e-6)

    # Check output shape, finite loss, and gradient flow.
    model.train()
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    logits, loss = model(first, targets)
    assert logits.shape == (1, 8, 64), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None and torch.isfinite(loss), f"Invalid loss: {loss}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "Missing or non-finite gradients"
    print("NYXVAULT transformer smoke tests: PASS (causal mask, shapes, loss, gradients)")

if __name__ == "__main__":
    main()
