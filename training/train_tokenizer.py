#!/usr/bin/env python3
"""Train a fresh BPE tokenizer from the prepared NYXVAULT corpus."""
import argparse, json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="training_data/train.txt")
    p.add_argument("--output",default="artifacts/tokenizer.json")
    p.add_argument("--vocab-size",type=int,default=12000)
    a=p.parse_args()
    src=Path(a.input)
    if not src.exists() or src.stat().st_size < 1000:
        raise SystemExit("Corpus is missing or too small (<1 KB). Run prepare_data.py first.")
    tok=Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer=normalizers.NFKC()
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train([str(src)],trainer)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    tok.save(a.output)
    print(f"Tokenizer saved: {a.output}; vocab={tok.get_vocab_size()}")
if __name__=="__main__": main()
