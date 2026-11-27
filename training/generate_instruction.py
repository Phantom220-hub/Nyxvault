#!/usr/bin/env python3
"""Generate a response from an NYXVAULT instruction-SFT checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

try:
    from model import NYXVAULTGPT
    from data_schema import render_prompt
except ImportError:
    from training.model import NYXVAULTGPT
    from training.data_schema import render_prompt


def main():
    p = argparse.ArgumentParser(description="NYXVAULT instruction SFT inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="/content/nyxvault-output/tokenizer.json")
    p.add_argument("--task", required=True)
    p.add_argument("--language", default="python")
    p.add_argument("--topic", default="complete-programs")
    p.add_argument("--level", type=int, default=8)
    p.add_argument("--type", default="write_code",
                   choices=("explain", "write_code", "complete", "debug", "review", "refactor", "design", "analyze"))
    p.add_argument("--input", default="")
    p.add_argument("--max-new", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding")
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()
    if args.max_new < 1 or args.top_k < 1 or args.temperature < 0:
        p.error("max-new/top-k must be positive; temperature cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != int(ckpt["vocab_size"]):
        raise ValueError("Tokenizer vocabulary does not match checkpoint.")
    model = NYXVAULTGPT(int(ckpt["vocab_size"]), ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rec = {"level": args.level, "topic": args.topic, "type": args.type,
           "language": args.language, "instruction": args.task}
    if args.input:
        rec["input"] = args.input
    prompt = render_prompt(rec)
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Prompt tokenization returned no tokens.")
    eos_id = tokenizer.token_to_id("[EOS]")
    generated = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(args.max_new):
            context = generated[:, -model.cfg["block_size"]:]
            logits, _ = model(context)
            next_logits = logits[:, -1, :].float()
            if args.temperature == 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / args.temperature
                if args.top_k < next_logits.shape[-1]:
                    values, _ = torch.topk(next_logits, args.top_k)
                    cutoff = values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            if eos_id is not None and int(next_id.item()) == eos_id:
                break

    answer = generated[0, len(ids):].tolist()
    if eos_id is not None and eos_id in answer:
        answer = answer[:answer.index(eos_id)]
    print(tokenizer.decode(answer).strip())


if __name__ == "__main__":
    main()
