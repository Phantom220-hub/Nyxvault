#!/usr/bin/env python3
"""Prepare NYXVAULT training text from legacy Q/A and structured programming JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from data_schema import normalize_record, render_record, validate_record
except ImportError:
    from training.data_schema import normalize_record, render_record, validate_record

FENCE = chr(96) * 3


def clean(text: Any) -> str:
    """Normalize line endings while preserving indentation, URLs and code contents."""
    value = str(text).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in value.split("\n")]
    normalized: List[str] = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                normalized.append("")
        else:
            blank_count = 0
            normalized.append(line)
    while normalized and not normalized[0].strip():
        normalized.pop(0)
    while normalized and not normalized[-1].strip():
        normalized.pop()
    return "\n".join(normalized)


def _digest(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def add_doc(
    docs: List[Dict[str, str]], seen: set[str], text: str, source: str, min_chars: int
) -> str:
    """Add one document and return 'added', 'short', or 'duplicate'."""
    text = clean(text)
    if len(text) < min_chars:
        return "short"
    digest = _digest(text)
    if digest in seen:
        return "duplicate"
    seen.add(digest)
    docs.append({"text": text, "source": source})
    return "added"


def _structured_text(item: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    record = normalize_record(item)
    issues = validate_record(record)
    blockers = [issue for issue in issues if issue.severity == "error" or issue.code == "W010"]
    if blockers:
        return None, [f"{issue.code}: {issue.message}" for issue in blockers]

    rendered = render_record(record)
    tests = record.get("tests", [])
    if tests and record.get("language") == "python":
        rendered += "\n\n### Verification tests\n" + FENCE + "python\n"
        rendered += "\n".join(tests) + "\n" + FENCE
    warnings = [f"{issue.code}: {issue.message}" for issue in issues if issue.severity == "warning"]
    return clean(rendered), warnings


def record_to_text(item: Any) -> Tuple[Optional[str], str, List[str]]:
    """Support both original question/answer records and the new programming schema."""
    if not isinstance(item, dict):
        return None, "unknown", ["record must be a JSON object"]

    structured_markers = {
        "id", "level", "language", "topic", "type", "instruction", "output", "license"
    }
    if structured_markers.intersection(item):
        rendered, issues = _structured_text(item)
        return rendered, "structured", issues

    if "question" in item or "answer" in item:
        question, answer = item.get("question"), item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            return None, "legacy", ["legacy record requires string question and answer fields"]
        question, answer = clean(question), clean(answer)
        if not question or not answer:
            return None, "legacy", ["legacy question and answer must not be empty"]
        return f"Вопрос: {question}\nОтвет: {answer}", "legacy", []

    return None, "unknown", [
        "expected structured fields or legacy question/answer fields"
    ]


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Optional[Any], Optional[str]]]:
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if line_no == 1 and raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            if not raw.strip():
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                yield line_no, None, f"invalid UTF-8: {exc.reason}"
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid JSON: {exc.msg} (column {exc.colno})"


def build_dataset(
    input_dir: str | Path,
    knowledge_file: str | Path,
    output_dir: str | Path,
    *,
    val_ratio: float = 0.05,
    min_chars: int = 80,
    seed: int = 42,
    min_documents: int = 10,
) -> Dict[str, Any]:
    """Create train.txt, validation.txt and a source/quality manifest."""
    if not 0.01 <= val_ratio <= 0.20:
        raise ValueError("val_ratio must be between 0.01 and 0.20")
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    if min_documents < 2:
        raise ValueError("min_documents must be at least 2")

    root = Path(input_dir)
    docs: List[Dict[str, str]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    issues: List[Dict[str, Any]] = []
    malformed = invalid = duplicates = too_short = 0

    def process_jsonl(path: Path, prefix: str) -> None:
        nonlocal malformed, invalid, duplicates, too_short
        for line_no, item, parse_error in _iter_jsonl(path):
            location = f"{path}:{line_no}"
            if parse_error:
                malformed += 1
                if len(issues) < 1000:
                    issues.append({"location": location, "reason": parse_error})
                continue

            rendered, fmt, warnings = record_to_text(item)
            format_counts[fmt] += 1
            if rendered is None:
                invalid += 1
                if len(issues) < 1000:
                    issues.append({"location": location, "reason": "; ".join(warnings)})
                continue
            if warnings and len(issues) < 1000:
                issues.append({"location": location, "warnings": warnings})

            min_record_chars = 20 if fmt == "legacy" else min_chars
            status = add_doc(docs, seen, rendered, f"{prefix}_{fmt}", min_record_chars)
            if status == "added":
                source_counts[f"{prefix}_{fmt}"] += 1
            elif status == "duplicate":
                duplicates += 1
            else:
                too_short += 1

    if root.exists():
        for file in sorted(root.rglob("*.txt")):
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                malformed += 1
                if len(issues) < 1000:
                    issues.append({"location": str(file), "reason": f"invalid UTF-8: {exc}"})
                continue
            status = add_doc(docs, seen, text, "raw_txt", min_chars)
            if status == "added":
                source_counts["raw_txt"] += 1
            elif status == "duplicate":
                duplicates += 1
            else:
                too_short += 1
        for file in sorted(root.rglob("*.jsonl")):
            process_jsonl(file, "raw_jsonl")

    knowledge = Path(knowledge_file)
    if knowledge.exists():
        process_jsonl(knowledge, "knowledge_jsonl")

    if len(docs) < min_documents:
        raise ValueError(
            f"Only {len(docs)} usable documents; at least {min_documents} are required. "
            "Add more valid, licensed or user-owned data."
        )

    random.Random(seed).shuffle(docs)
    val_count = max(1, int(len(docs) * val_ratio))
    validation_docs, train_docs = docs[:val_count], docs[val_count:]
    if not train_docs:
        raise ValueError("Dataset split produced an empty training set.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_text = "\n\n".join(item["text"] for item in train_docs)
    validation_text = "\n\n".join(item["text"] for item in validation_docs)
    (output / "train.txt").write_text(train_text, encoding="utf-8")
    (output / "validation.txt").write_text(validation_text, encoding="utf-8")

    manifest: Dict[str, Any] = {
        "documents": len(docs),
        "train_documents": len(train_docs),
        "validation_documents": len(validation_docs),
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "sources": dict(source_counts),
        "formats_seen": dict(format_counts),
        "malformed_jsonl_lines_or_text_files": malformed,
        "invalid_jsonl_records": invalid,
        "duplicate_records": duplicates,
        "too_short_records": too_short,
        "seed": seed,
        "validation_ratio": val_ratio,
        "issues_sample": issues,
        "note": "Document-level split after normalized deduplication. Review data licenses and topic overlap.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NYXVAULT training data from legacy and structured JSONL."
    )
    parser.add_argument("--input", default="training_data/raw")
    parser.add_argument("--knowledge", default="data/knowledge.jsonl")
    parser.add_argument("--out", default="training_data")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--min-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = build_dataset(
            args.input, args.knowledge, args.out,
            val_ratio=args.val_ratio,
            min_chars=args.min_chars,
            seed=args.seed,
            min_documents=args.min_documents,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
