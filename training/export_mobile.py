#!/usr/bin/env python3
"""Export the trained NYXVAULT Transformer and its ByteLevel BPE tokenizer for Android."""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import NYXVAULTGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--out", default="mobile/nyxvault.onnx")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NYXVAULTGPT(checkpoint["vocab_size"], checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    class Export(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids):
            return self.wrapped(input_ids)[0]

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Export(model),
        torch.ones((1, 16), dtype=torch.long),
        str(target),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer_data = json.loads(tokenizer.to_str())
    bpe = tokenizer_data.get("model", {})
    if bpe.get("type") != "BPE" or not bpe.get("vocab") or not bpe.get("merges"):
        raise SystemExit("Android export requires the project's ByteLevel BPE tokenizer.")

    merges = [
        " ".join(item) if isinstance(item, list) else str(item)
        for item in bpe["merges"]
    ]
    mobile_tokenizer = {
        "vocab": bpe["vocab"],
        "merges": merges,
        "unk_token": bpe.get("unk_token", "[UNK]"),
        "eos_token": "[EOS]",
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel",
    }
    tokenizer_out = target.with_name("tokenizer_mobile.json")
    tokenizer_out.write_text(
        json.dumps(mobile_tokenizer, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantized = target.with_name(target.stem + "_int8.onnx")
        quantize_dynamic(str(target), str(quantized), weight_type=QuantType.QInt8)
        print("Quantized:", quantized, "size_MB", round(quantized.stat().st_size / 1e6, 2))

    print("Exported ONNX:", target, "size_MB", round(target.stat().st_size / 1e6, 2))
    print("Exported Android tokenizer:", tokenizer_out)


if __name__ == "__main__":
    main()
