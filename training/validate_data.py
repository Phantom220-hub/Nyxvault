#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NYXVAULT: валидатор обучающих данных по программированию (JSONL).

Проверяет структуру записей (см. data_schema.py), синтаксис Python/JSON в блоках кода,
дубликаты, заглушки, покрытие учебной программы (curriculum.py) и печатает отчёт.
Опционально выполняет поле "tests" и записывает очищенный файл.

Примеры:
    python training/validate_data.py data/programming_seed.jsonl
    python training/validate_data.py data/ --strict --report-json reports/data.json
    python training/validate_data.py data/raw.jsonl --clean-out data/clean.jsonl
    python training/validate_data.py data/raw.jsonl --exec-tests --timeout 10

Коды выхода: 0 нет ошибок, 1 есть ошибки, 2 нет входных файлов или не удалось записать результат.

Важно про --exec-tests: код из записей запускается отдельным процессом Python с лимитом
времени (и лимитами CPU/памяти на Linux/macOS), но это НЕ полноценная песочница.
Используйте флаг только для данных, которым вы доверяете, лучше в Colab или контейнере.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from curriculum import LEVELS
    from data_schema import (
        Issue,
        canonical_language,
        canonical_record,
        check_code_block,
        extract_code_blocks,
        normalize_record,
        normalize_text,
        render_record,
        validate_record,
    )
except ImportError:  # запуск как пакет: python -m training.validate_data
    from training.curriculum import LEVELS
    from training.data_schema import (
        Issue,
        canonical_language,
        canonical_record,
        check_code_block,
        extract_code_blocks,
        normalize_record,
        normalize_text,
        render_record,
        validate_record,
    )

CHARS_PER_TOKEN_ESTIMATE = 3.5  # грубая оценка; реальное число зависит от вашего BPE-токенизатора


# ---------------------------------------------------------------------------
# Чтение файлов
# ---------------------------------------------------------------------------

def collect_files(paths: List[str]) -> Tuple[List[str], List[str]]:
    """Возвращает (список .jsonl файлов, список сообщений об отсутствующих путях)."""
    files: List[str] = []
    missing: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if name.endswith(".jsonl"):
                        files.append(os.path.join(root, name))
        elif os.path.isfile(path):
            files.append(path)
        else:
            missing.append(path)
    return sorted(dict.fromkeys(files)), missing


def read_jsonl(path: str) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    """Генерирует (номер_строки, объект, ошибка_разбора). Пустые строки пропускаются."""
    with open(path, "rb") as handle:
        for lineno, raw in enumerate(handle, 1):
            if lineno == 1 and raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            if not raw.strip():
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                yield lineno, None, f"строка не в кодировке UTF-8: {exc.reason}"
                continue
            try:
                yield lineno, json.loads(text), None
            except json.JSONDecodeError as exc:
                yield lineno, None, f"некорректный JSON: {exc.msg} (столбец {exc.colno})"


# ---------------------------------------------------------------------------
# Дубликаты
# ---------------------------------------------------------------------------

def _digest(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def task_key(rec: Dict[str, Any]) -> str:
    return _digest(rec["instruction"] + "\n" + rec.get("input", ""))


def output_key(rec: Dict[str, Any]) -> str:
    return _digest(rec["output"])

def normalize_legacy_record(obj: Any) -> Tuple[Optional[Dict[str, str]], List[Issue]]:
    """Validate the repository's pre-schema question/answer JSONL records."""
    if not isinstance(obj, dict):
        return None, [Issue("error", "E050", "legacy record must be a JSON object")]
    question, answer = obj.get("question"), obj.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None, [Issue("error", "E050", "legacy record requires string question and answer fields")]
    question, answer = normalize_text(question), normalize_text(answer)
    if not question or not answer:
        return None, [Issue("error", "E050", "legacy question and answer must not be empty")]

    record = {"question": question, "answer": answer}
    issues: List[Issue] = []
    source = obj.get("source")
    if source is not None:
        if not isinstance(source, str):
            issues.append(Issue("error", "E050", "legacy source must be a string when provided"))
        elif source.strip():
            record["source"] = source.strip()

    blocks, balanced = extract_code_blocks(answer)
    if not balanced:
        issues.append(Issue("error", "E030", "legacy answer contains an unclosed code block"))
    for tag, code, start in blocks:
        language = canonical_language(tag) if tag else ""
        code_issue = check_code_block(language, code) if language else None
        if code_issue and code_issue.severity == "error":
            issues.append(Issue("error", code_issue.code, f"{code_issue.message} [block at line {start}]"))
    return record, issues


def is_legacy_record(obj: Any) -> bool:
    """Return True for old question/answer records that are not new-schema records."""
    if not isinstance(obj, dict):
        return False
    structured_markers = {"id", "level", "language", "topic", "type", "instruction", "output", "license"}
    return not structured_markers.intersection(obj) and ("question" in obj or "answer" in obj)


# ---------------------------------------------------------------------------
# Выполнение tests (опционально)
# ---------------------------------------------------------------------------

_RUNNER = (
    "import runpy, sys; "
    "runpy.run_path(sys.argv[1], run_name='nyx_case')"
)


def _limit_resources(timeout: float) -> None:  # pragma: no cover (выполняется в дочернем процессе)
    try:
        import resource
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        memory = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    except Exception:
        pass


def run_python_tests(rec: Dict[str, Any], timeout: float) -> Optional[Issue]:
    """Запускает код из python-блоков output вместе с tests. None если всё прошло."""
    blocks, _ = extract_code_blocks(rec["output"])
    code = "\n\n".join(
        block for tag, block, _line in blocks
        if (canonical_language(tag) if tag else rec["language"]) == "python"
    )
    if not code.strip():
        return Issue("error", "E035", "заданы tests, но в output нет блока Python для проверки")
    program = code + "\n\n" + "\n".join(rec["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "case.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(program)
        kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = lambda: _limit_resources(timeout)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER, script],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return Issue("error", "E036", f"tests не уложились в {timeout:g} с")
        except OSError as exc:
            return Issue("error", "E036", f"не удалось запустить tests: {exc}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        message = tail[-1] if tail else f"код возврата {proc.returncode}"
        return Issue("error", "E036", f"tests не прошли: {message[:300]}")
    return None


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Валидатор JSONL-данных NYXVAULT для обучения программированию."
    )
    parser.add_argument("paths", nargs="+", help="файлы .jsonl или папки с ними")
    parser.add_argument("--strict", action="store_true", help="считать предупреждения ошибками")
    parser.add_argument("--exec-tests", action="store_true", help="выполнять поле tests (только для доверенных данных)")
    parser.add_argument("--timeout", type=float, default=10.0, help="лимит времени на запись при --exec-tests, секунд")
    parser.add_argument("--clean-out", metavar="FILE", help="записать валидные записи (нормализованные) в этот файл")
    parser.add_argument("--report-json", metavar="FILE", help="сохранить отчёт в JSON")
    parser.add_argument("--max-show", type=int, default=30, help="сколько проблем печатать (по умолчанию 30)")
    return parser


def _make_dirs_for(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    files, missing = collect_files(args.paths)
    for path in missing:
        print(f"Путь не найден: {path}", file=sys.stderr)
    if not files:
        print("Нет входных .jsonl файлов.", file=sys.stderr)
        return 2
    if args.clean_out and os.path.abspath(args.clean_out) in {os.path.abspath(f) for f in files}:
        print("--clean-out не должен совпадать с входным файлом.", file=sys.stderr)
        return 2

    issues_log: List[Tuple[str, Issue]] = []
    issue_counts: Counter = Counter()
    total_records = 0
    valid_records: List[Dict[str, Any]] = []  # structured records
    valid_legacy_records: List[Dict[str, str]] = []
    clean_records: List[Dict[str, Any]] = []
    legacy_task_seen: Dict[str, str] = {}
    legacy_chars = 0

    seen_task: Dict[str, str] = {}
    seen_id: Dict[str, str] = {}
    seen_output: Dict[str, str] = {}

    def add(location: str, issue: Issue) -> bool:
        """Регистрирует проблему, возвращает True если она блокирующая."""
        # Known code placeholders are not accepted into the training corpus.
        blocking = issue.severity == "error" or issue.code == "W010" or args.strict
        issues_log.append((location, issue))
        issue_counts[(("error" if blocking else "warning"), issue.code)] += 1
        return blocking

    for path in files:
        for lineno, obj, parse_error in read_jsonl(path):
            total_records += 1
            location = f"{path}:{lineno}"
            if parse_error:
                add(location, Issue("error", "E000", parse_error))
                continue

            if is_legacy_record(obj):
                legacy, legacy_issues = normalize_legacy_record(obj)
                blocking = False
                for issue in legacy_issues:
                    blocking |= add(location, issue)
                if legacy is None or any(issue.severity == "error" for issue in legacy_issues):
                    continue
                key = _digest(legacy["question"])
                if key in legacy_task_seen:
                    blocking |= add(location, Issue("error", "E042", f"duplicate legacy question (first seen: {legacy_task_seen[key]})"))
                else:
                    legacy_task_seen[key] = location
                if not blocking:
                    valid_legacy_records.append(legacy)
                    clean_records.append(legacy)
                    legacy_chars += len(legacy["question"]) + len(legacy["answer"])
                continue

            rec = normalize_record(obj)
            blocking = False
            validation_issues = validate_record(rec)
            for issue in validation_issues:
                blocking |= add(location, issue)
            if not isinstance(rec, dict) or any(
                issue.code in ("E001", "E002", "E003") for issue in validation_issues
            ):
                continue

            # Дубликаты проверяем только у структурно корректных записей.
            if rec["id"] in seen_id:
                blocking |= add(location, Issue("error", "E041", f"повторяющийся id '{rec['id']}' (первый раз: {seen_id[rec['id']]})"))
            else:
                seen_id[rec["id"]] = location
            tkey = task_key(rec)
            if tkey in seen_task:
                blocking |= add(location, Issue("error", "E040", f"дубликат задания (первый раз: {seen_task[tkey]})"))
            else:
                seen_task[tkey] = location
                okey = output_key(rec)
                if okey in seen_output:
                    blocking |= add(location, Issue("warning", "W040", f"такой же output уже есть (первый раз: {seen_output[okey]})"))
                else:
                    seen_output[okey] = location

            if args.exec_tests and not blocking and "tests" in rec and rec["language"] == "python":
                failure = run_python_tests(rec, args.timeout)
                if failure:
                    blocking |= add(location, failure)

            if not blocking:
                valid_records.append(rec)
                clean_records.append(canonical_record(rec))

    # --- статистика по валидным записям ---
    by_level: Counter = Counter()
    by_language: Counter = Counter()
    by_type: Counter = Counter()
    by_topic: Counter = Counter()
    chars_by_level: Counter = Counter()
    for rec in valid_records:
        by_level[rec["level"]] += 1
        by_language[rec["language"]] += 1
        by_type[rec["type"]] += 1
        by_topic[(rec["level"], rec["topic"])] += 1
        chars_by_level[rec["level"]] += len(render_record(rec))

    error_count = sum(n for (sev, _code), n in issue_counts.items() if sev == "error")
    warning_count = sum(n for (sev, _code), n in issue_counts.items() if sev == "warning")
    total_chars = sum(chars_by_level.values())

    # --- вывод ---
    valid_total = len(valid_records) + len(valid_legacy_records)
    print(
        f"Файлов: {len(files)}   записей: {total_records}   валидных: {valid_total} "
        f"(structured: {len(valid_records)}, legacy: {len(valid_legacy_records)})"
    )
    print(f"Ошибок: {error_count}   предупреждений: {warning_count}" + ("   (режим --strict)" if args.strict else ""))
    if issues_log:
        print("\nПроблемы (первые %d):" % min(args.max_show, len(issues_log)))
        for location, issue in issues_log[: args.max_show]:
            label = "ОШИБКА" if (issue.severity == "error" or args.strict) else "предупр."
            print(f"  {label:8} {location}  [{issue.code}] {issue.message}")
        if len(issues_log) > args.max_show:
            print(f"  ... и ещё {len(issues_log) - args.max_show} (см. --report-json или увеличьте --max-show)")
        print("\nПо кодам:")
        for (severity, code), count in sorted(issue_counts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"  {code} ({severity}): {count}")

    print("\nПокрытие учебной программы (только валидные записи):")
    coverage: List[Dict[str, Any]] = []
    for level in LEVELS:
        covered = 0
        gaps: List[str] = []
        for topic in level.topics:
            count = by_topic[(level.id, topic)]
            if count >= level.min_examples:
                covered += 1
            else:
                gaps.append(f"{topic} ({count}/{level.min_examples})")
        tokens = int(chars_by_level[level.id] / CHARS_PER_TOKEN_ESTIMATE)
        share = chars_by_level[level.id] / total_chars if total_chars else 0.0
        coverage.append({
            "level": level.id, "name": level.name, "records": by_level[level.id],
            "approx_tokens": tokens, "share": round(share, 4), "target_share": level.token_share,
            "topics_covered": covered, "topics_total": len(level.topics), "gaps": gaps,
        })
        print(
            f"  Уровень {level.id} {level.name}: записей {by_level[level.id]}, ~{tokens} ток., "
            f"доля {share:.0%} (цель {level.token_share:.0%}), тем покрыто {covered}/{len(level.topics)}"
        )
        if gaps:
            shown = ", ".join(gaps[:6]) + (f" и ещё {len(gaps) - 6}" if len(gaps) > 6 else "")
            print(f"      пробелы: {shown}")

    print(f"\nПо языкам: {dict(by_language.most_common())}")
    print(f"По типам задач: {dict(by_type.most_common())}")
    print(
        f"Оценка объёма: ~{int(total_chars / CHARS_PER_TOKEN_ESTIMATE)} токенов "
        f"(грубо, {CHARS_PER_TOKEN_ESTIMATE} символа на токен; точное число даст ваш токенизатор)"
    )

    if args.clean_out:
        try:
            _make_dirs_for(args.clean_out)
            with open(args.clean_out, "w", encoding="utf-8", newline="\n") as handle:
                for rec in clean_records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"\nОчищенный файл записан: {args.clean_out} ({valid_total} записей)")
        except OSError as exc:
            print(f"Не удалось записать {args.clean_out}: {exc}", file=sys.stderr)
            return 2

    if args.report_json:
        report = {
            "files": files,
            "total_records": total_records,
            "valid_records": valid_total,
            "valid_structured_records": len(valid_records),
            "valid_legacy_records": len(valid_legacy_records),
            "errors": error_count,
            "warnings": warning_count,
            "strict": args.strict,
            "issue_counts": [
                {"severity": sev, "code": code, "count": count}
                for (sev, code), count in sorted(issue_counts.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            ],
            "issues": [
                {"location": loc, "severity": i.severity, "code": i.code, "message": i.message}
                for loc, i in issues_log[:1000]
            ],
            "by_level": {str(k): v for k, v in sorted(by_level.items())},
            "by_language": dict(by_language),
            "by_type": dict(by_type),
            "coverage": coverage,
            "approx_tokens": int((total_chars + legacy_chars) / CHARS_PER_TOKEN_ESTIMATE),
        }
        try:
            _make_dirs_for(args.report_json)
            with open(args.report_json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(f"Отчёт записан: {args.report_json}")
        except OSError as exc:
            print(f"Не удалось записать {args.report_json}: {exc}", file=sys.stderr)
            return 2

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())