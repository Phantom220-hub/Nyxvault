#!/usr/bin/env python3
"""Build deterministic train/validation text files from licensed or user-owned data."""
import argparse, hashlib, json, random, re
from collections import Counter
from pathlib import Path


def clean(s):
    s = str(s).replace("\\x00", "").replace("\\r", "\\n")
    s = re.sub(r"https?://\\S+", " ", s)
    s = re.sub(r"[ \\t]+", " ", s)
    s = re.sub(r"\\n{3,}", "\\n\\n", s)
    return s.strip()


def add_doc(docs, seen, text, source, min_chars):
    text = clean(text)
    if len(text) < min_chars:
        return False
    digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
    if digest in seen:
        return False
    seen.add(digest)
    docs.append({"text": text, "source": source})
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="training_data/raw")
    p.add_argument("--knowledge", default="data/knowledge.jsonl")
    p.add_argument("--out", default="training_data")
    p.add_argument("--val-ratio", type=float, default=.05)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not 0.01 <= a.val_ratio <= 0.2:
        raise SystemExit("--val-ratio must be between 0.01 and 0.20")
    root = Path(a.input)
    docs, seen = [], set()
    counts = Counter()
    malformed = 0
    if root.exists():
        for f in sorted(root.rglob("*.txt")):
            if add_doc(docs, seen, f.read_text(encoding="utf-8", errors="ignore"), "raw_txt", a.min_chars):
                counts["raw_txt"] += 1
        for f in sorted(root.rglob("*.jsonl")):
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    item = json.loads(line)
                    q, ans = clean(item.get("question", "")), clean(item.get("answer", ""))
                    if q and ans and add_doc(docs, seen, f"Вопрос: {q}\\nОтвет: {ans}", "raw_jsonl", 20):
                        counts["raw_jsonl"] += 1
                except (json.JSONDecodeError, AttributeError, TypeError):
                    malformed += 1

    knowledge = Path(a.knowledge)
    if knowledge.exists():
        for line in knowledge.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                q, ans = clean(item.get("question", "")), clean(item.get("answer", ""))
                if not q or not ans:
                    counts["knowledge_empty"] += 1
                    continue
                if add_doc(docs, seen, f"Вопрос: {q}\\nОтвет: {ans}", "knowledge_jsonl", 20):
                    counts["knowledge_jsonl"] += 1
                else:
                    counts["duplicates_or_short"] += 1
            except (json.JSONDecodeError, AttributeError, TypeError):
                malformed += 1
    if len(docs) < 10:
        raise SystemExit(f"Only {len(docs)} usable documents. Add more licensed/user-owned text.")
    random.Random(a.seed).shuffle(docs)
    n_val = max(1, int(len(docs) * a.val_ratio))
    val, train = docs[:n_val], docs[n_val:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_text = "\\n\\n".join(x["text"] for x in train)
    val_text = "\\n\\n".join(x["text"] for x in val)
    (out / "train.txt").write_text(train_text, encoding="utf-8")
    (out / "validation.txt").write_text(val_text, encoding="utf-8")
    manifest = {
        "documents": len(docs),
        "train_documents": len(train),
        "validation_documents": len(val),
        "train_characters": len(train_text),
        "validation_characters": len(val_text),
        "sources": dict(counts),
        "malformed_jsonl_lines": malformed,
        "seed": a.seed,
        "validation_ratio": a.val_ratio,
        "note": "Document-level random split; inspect topic overlap before interpreting validation metrics."
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
