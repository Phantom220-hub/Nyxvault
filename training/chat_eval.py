#!/usr/bin/env python3
"""Run deterministic Russian chat/code prompts against the current ONNX model."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


WORD_RE = re.compile(r"[\w]+", re.UNICODE)
CHECK_WEIGHTS = {
    "non_empty": 10,
    "no_prompt_template_leak": 10,
    "not_obviously_repetitive": 10,
    "minimum_answer_length": 10,
    "contains_expected_concept_words": 60,
}


def score_answer(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    """Score content requirements much more heavily than superficial formatting."""
    text = answer.strip()
    folded = text.casefold()
    words = WORD_RE.findall(folded)
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    expected_groups = case.get("must_include_any", [])
    missing_groups = [
        group for group in expected_groups
        if not any(str(term).casefold() in folded for term in group)
    ]
    checks = {
        "non_empty": bool(text),
        "no_prompt_template_leak": not any(
            marker in text for marker in ("### Task", "### Solution", "### Level:")
        ),
        "not_obviously_repetitive": (
            unique_ratio >= 0.30 if len(words) > 4 else len(words) > 0
        ) and not re.search(r"\b(\w+)(?:\s+\1){3,}\b", text, re.IGNORECASE),
        "minimum_answer_length": len(words) >= int(case.get("min_words", 3)),
        "contains_expected_concept_words": not missing_groups,
    }
    points = sum(CHECK_WEIGHTS[name] for name, passed in checks.items() if passed)
    return {
        "checks": checks,
        "heuristic_score": points,
        "word_count": len(words),
        "unique_word_ratio": round(unique_ratio, 3),
        "missing_expected_groups": missing_groups,
    }


def generate(
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 12,
) -> tuple[str, int]:
    ids = tokenizer.encode(prompt).ids[-512:]
    if not ids:
        return "", 0
    generated: list[int] = []
    eos_id = tokenizer.token_to_id("[EOS]")
    rng = random.Random(seed)

    for _ in range(max_new_tokens):
        context = np.asarray([ids[-512:]], dtype=np.int64)
        logits = session.run(["logits"], {"input_ids": context})[0][0, -1]
        k = min(top_k, len(logits))
        candidates = np.argpartition(logits, -k)[-k:]
        candidates = sorted(candidates.tolist(), key=lambda idx: float(logits[idx]), reverse=True)
        scaled = [float(logits[idx]) / max(temperature, 0.05) for idx in candidates]
        peak = max(scaled)
        weights = [math.exp(value - peak) for value in scaled]
        draw = rng.random() * sum(weights)
        next_id = candidates[-1]
        for candidate, weight in zip(candidates, weights):
            draw -= weight
            if draw <= 0:
                next_id = candidate
                break
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
                raise ValueError(f"{path}:{line_number}: each case needs id and prompt")
            if not isinstance(item.get("must_include_any", []), list):
                raise ValueError(f"{path}:{line_number}: must_include_any must be a list")
            if int(item.get("min_words", 3)) < 1:
                raise ValueError(f"{path}:{line_number}: min_words must be positive")
            cases.append(item)
    if not cases:
        raise ValueError("No evaluation cases were found")
    return cases


def run_evaluation(
    model_path: str | Path,
    tokenizer_path: str | Path,
    cases_path: str | Path,
    *,
    seed: int = 20261128,
    max_new_tokens: int = 120,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    case_results = []
    for index, case in enumerate(load_cases(cases_path)):
        answer, generated_tokens = generate(
            session, tokenizer, case["prompt"],
            seed=seed + index, max_new_tokens=max_new_tokens,
        )
        scored = score_answer(answer, case)
        result = {
            "id": case["id"],
            "category": case.get("category", "unspecified"),
            "prompt": case["prompt"],
            "answer": answer,
            "generated_tokens": generated_tokens,
            **scored,
        }
        case_results.append(result)
        print(f"\n===== {case['id']} | score={scored['heuristic_score']}/100 =====")
        print("USER PROMPT:", case["prompt"].split("### Solution")[0].split("### Task\n")[-1].strip())
        print("NYXVAULT:", answer if answer else "[EMPTY RESPONSE]")
        print("CHECKS:", json.dumps(scored["checks"], ensure_ascii=False))

    total = len(case_results)
    summary = {
        "cases": total,
        "mean_heuristic_score": round(
            sum(item["heuristic_score"] for item in case_results) / total, 1
        ),
        "non_empty_answers": sum(bool(item["answer"]) for item in case_results),
        "content_requirement_passes": sum(
            item["checks"]["contains_expected_concept_words"] for item in case_results
        ),
        "minimum_length_passes": sum(
            item["checks"]["minimum_answer_length"] for item in case_results
        ),
        "clean_format_answers": sum(
            item["checks"]["no_prompt_template_leak"] for item in case_results
        ),
        "low_repetition_answers": sum(
            item["checks"]["not_obviously_repetitive"] for item in case_results
        ),
    }
    return {
        "format": 2,
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "seed": seed,
        "scoring_weights": CHECK_WEIGHTS,
        "scoring_note": (
            "Most points require the expected concept terms. This remains a simple "
            "keyword/format smoke benchmark, not a proof of semantic correctness; "
            "human review and reference-answer tests are still required."
        ),
        "summary": summary,
        "results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cases", default="training/chat_eval_cases.jsonl")
    parser.add_argument("--output", default="artifacts/chat_eval_report.json")
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    report = run_evaluation(
        args.model, args.tokenizer, args.cases,
        seed=args.seed, max_new_tokens=args.max_new_tokens,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("Full transcript report:", out)


if __name__ == "__main__":
    main()
