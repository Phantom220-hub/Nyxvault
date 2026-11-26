#!/usr/bin/env python3
"""End-to-end smoke test for ONNX and tokenizer export consumed by Android."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from model import NYXVAULTGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    cfg = {"d_model": 32, "n_heads": 4, "n_layers": 1, "block_size": 32, "dropout": 0.0}

    with tempfile.TemporaryDirectory(prefix="nyxvault-export-") as tmp:
        root = Path(tmp)
        checkpoint_path = root / "tiny_checkpoint.pt"
        model_path = root / "nyxvault.onnx"
        model = NYXVAULTGPT(vocab_size, cfg)
        torch.save({
            "model": model.state_dict(),
            "config": cfg,
            "vocab_size": vocab_size,
            "step": 0,
            "best_val": 0.0,
        }, checkpoint_path)

        subprocess.run([
            sys.executable, "training/export_mobile.py",
            "--checkpoint", str(checkpoint_path),
            "--tokenizer", str(tokenizer_path),
            "--out", str(model_path),
        ], check=True)

        mobile_tokenizer_path = root / "tokenizer_mobile.json"
        assert model_path.is_file() and model_path.stat().st_size > 0
        assert mobile_tokenizer_path.is_file() and mobile_tokenizer_path.stat().st_size > 0
        mobile = json.loads(mobile_tokenizer_path.read_text(encoding="utf-8"))
        assert len(mobile["vocab"]) == vocab_size
        assert all(isinstance(pair, str) for pair in mobile["merges"])

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_ids = np.full((1, 8), 1, dtype=np.int64)
        logits = session.run(["logits"], {"input_ids": input_ids})[0]
        assert logits.shape == (1, 8, vocab_size), logits.shape
        assert np.isfinite(logits).all()
        print("NYXVAULT Android export smoke test: PASS")


if __name__ == "__main__":
    main()
