# -*- coding: utf-8 -*-
"""NYXVAULT: схема обучающих записей по программированию (JSONL, одна запись на строку).

Формат записи (поля, помеченные *, обязательны):

    id*          уникальный идентификатор: строчные латинские буквы, цифры, _ . -  (3..64 символа)
    level*       целое число, уровень учебной программы (см. curriculum.py)
    language*    язык программирования или русский язык из LANGUAGES
    topic*       тема из curriculum.py, принадлежащая указанному уровню
    type*        вид задачи из RECORD_TYPES
    instruction* текст задания
    output*      эталонный ответ; код оформляется блоками ```язык ... ```
    license*     происхождение/лицензия из LICENSES
    input        исходный код или данные к заданию (обязателен для debug, review,
                 refactor, analyze, complete)
    tests        список Python-операторов (обычно assert) для проверки кода из output
    tags         список строк
    source       свободное описание происхождения

Модуль не зависит от архитектуры модели и токенизатора: запись превращается в обычный
текст функцией render_record(), а как этот текст токенизируется, решает prepare_data.py.

Только стандартная библиотека Python.
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from curriculum import LEVELS, TOPIC_TO_LEVEL, topics_for_level
except ImportError:  # запуск как пакет: python -m training.validate_data
    from training.curriculum import LEVELS, TOPIC_TO_LEVEL, topics_for_level

SCHEMA_VERSION = 1

LANGUAGES: Tuple[str, ...] = (
    "python", "ru", "javascript", "typescript", "java", "kotlin", "c", "cpp", "csharp",
    "go", "rust", "sql", "bash", "html", "css", "json", "gdscript",
)

RECORD_TYPES: Tuple[str, ...] = (
    "explain",      # объяснить понятие или код
    "dialogue",     # обычный диалог пользователя и ассистента
    "write_code",   # написать код по заданию
    "complete",     # дописать код (input содержит начало)
    "debug",        # найти и исправить ошибку (input содержит код с ошибкой)
    "review",       # провести ревью кода (input)
    "refactor",     # улучшить код без смены поведения (input)
    "design",       # спроектировать структуру программы или модуля
    "analyze",      # проанализировать код или проект (input)
)

# Для этих видов задач поле input обязательно.
TYPES_REQUIRING_INPUT = frozenset({"debug", "review", "refactor", "analyze", "complete"})
# Для этих видов задач в output ожидается хотя бы один блок кода.
TYPES_EXPECTING_CODE = frozenset({"write_code", "complete", "debug", "refactor"})

LICENSES: Tuple[str, ...] = (
    "original",          # написано владельцем проекта
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0", "public-domain",
)

REQUIRED_FIELDS: Tuple[str, ...] = (
    "id", "level", "language", "topic", "type", "instruction", "output", "license",
)
OPTIONAL_FIELDS: Tuple[str, ...] = ("input", "tests", "tags", "source")
ALL_FIELDS: Tuple[str, ...] = REQUIRED_FIELDS + OPTIONAL_FIELDS
TEXT_FIELDS: Tuple[str, ...] = ("instruction", "input", "output")

# (минимум, максимум) длины в символах после нормализации.
FIELD_LIMITS: Dict[str, Tuple[int, int]] = {
    "instruction": (1, 4000),
    "input": (1, 24000),
    "output": (1, 32000),
}

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]{2,63}$")

LANGUAGE_ALIASES: Dict[str, str] = {
    "py": "python", "python3": "python", "js": "javascript", "node": "javascript",
    "ts": "typescript", "c++": "cpp", "cc": "cpp", "c#": "csharp", "cs": "csharp",
    "kt": "kotlin", "sh": "bash", "shell": "bash", "zsh": "bash", "gd": "gdscript",
    "rs": "rust", "golang": "go",
}


@dataclass(frozen=True)
class Issue:
    severity: str   # "error" или "warning"
    code: str
    message: str


def _err(code: str, message: str) -> Issue:
    return Issue("error", code, message)


def _warn(code: str, message: str) -> Issue:
    return Issue("warning", code, message)


# ---------------------------------------------------------------------------
# Нормализация
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFC, переводы строк \\n, без пробелов в конце строк, без пустых строк по краям."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip("\n")


def normalize_record(rec: Any) -> Any:
    """Возвращает нормализованную копию записи; значения неверного типа не трогает."""
    if not isinstance(rec, dict):
        return rec
    out = dict(rec)
    for key in ("id", "language", "topic", "type", "license"):
        if isinstance(out.get(key), str):
            out[key] = out[key].strip().lower()
    for key in TEXT_FIELDS:
        if isinstance(out.get(key), str):
            out[key] = normalize_text(out[key])
    if isinstance(out.get("source"), str):
        out["source"] = out["source"].strip()
    if isinstance(out.get("tests"), list):
        out["tests"] = [normalize_text(t) if isinstance(t, str) else t for t in out["tests"]]
    if isinstance(out.get("tags"), list):
        out["tags"] = [t.strip().lower() if isinstance(t, str) else t for t in out["tags"]]
    return out


def canonical_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Запись в каноническом порядке полей; неизвестные поля отбрасываются."""
    return {key: rec[key] for key in ALL_FIELDS if key in rec}


def canonical_language(tag: str) -> str:
    tag = tag.strip().lower()
    return LANGUAGE_ALIASES.get(tag, tag)


# ---------------------------------------------------------------------------
# Блоки кода
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> Tuple[List[Tuple[str, str, int]], bool]:
    """Находит блоки ```язык ... ```.

    Возвращает ([(тег_языка, код, номер_строки_открытия), ...], все_блоки_закрыты).
    """
    blocks: List[Tuple[str, str, int]] = []
    in_block = False
    tag = ""
    start = 0
    buf: List[str] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_block:
                in_block = True
                rest = stripped[3:].strip()
                tag = rest.split()[0] if rest else ""
                start = lineno
                buf = []
            else:
                blocks.append((tag, "\n".join(buf), start))
                in_block = False
            continue
        if in_block:
            buf.append(line)
    return blocks, not in_block


# language -> (однострочный комментарий, блочные /* */, кавычка ', обратная кавычка `)
_BRACKET_SYNTAX: Dict[str, Tuple[str, bool, bool, bool]] = {
    "javascript": ("//", True, True, True),
    "typescript": ("//", True, True, True),
    "java": ("//", True, True, False),
    "csharp": ("//", True, True, False),
    "kotlin": ("//", True, True, False),
    "c": ("//", True, True, False),
    "cpp": ("//", True, True, False),
    "go": ("//", True, True, True),
    "rust": ("//", True, False, False),
    "gdscript": ("#", False, True, False),
}

_CLOSERS = {")": "(", "]": "[", "}": "{"}


def _bracket_problem(code: str, language: str) -> Optional[str]:
    """Эвристическая проверка баланса скобок. Строки и комментарии пропускаются."""
    line_comment, block_comments, single_quote, backtick = _BRACKET_SYNTAX[language]
    stack: List[Tuple[str, int]] = []
    i, n, line = 0, len(code), 1
    while i < n:
        ch = code[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if code.startswith(line_comment, i):
            while i < n and code[i] != "\n":
                i += 1
            continue
        if block_comments and code.startswith("/*", i):
            end = code.find("*/", i + 2)
            end = n if end == -1 else end + 2
            line += code.count("\n", i, end)
            i = end
            continue
        if ch == '"' or (ch == "'" and single_quote) or (ch == "`" and backtick):
            quote = ch
            i += 1
            while i < n:
                c = code[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "\n":
                    line += 1
                    if quote != "`":
                        break
                if c == quote:
                    break
                i += 1
            i += 1
            continue
        if ch in "([{":
            stack.append((ch, line))
        elif ch in _CLOSERS:
            if not stack or stack[-1][0] != _CLOSERS[ch]:
                return f"лишняя или несогласованная скобка '{ch}' (строка {line})"
            stack.pop()
        i += 1
    if stack:
        ch, opened = stack[-1]
        return f"не закрыта скобка '{ch}' (строка {opened})"
    return None


def check_code_block(language: str, code: str) -> Optional[Issue]:
    """Проверка одного блока кода. Для python и json ошибка критична, для остальных это предупреждение."""
    if language == "python":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return _err("E031", f"синтаксическая ошибка Python в блоке кода: {exc.msg} (строка {exc.lineno})")
        except (ValueError, RecursionError, MemoryError) as exc:
            return _err("E031", f"блок Python не удалось разобрать: {exc}")
        return None
    if language == "json":
        try:
            json.loads(code)
        except ValueError as exc:
            return _err("E032", f"некорректный JSON в блоке кода: {exc}")
        return None
    if language in _BRACKET_SYNTAX:
        problem = _bracket_problem(code, language)
        if problem:
            return _warn("W011", f"возможная ошибка скобок в блоке {language}: {problem}")
    return None


# Следы заглушек в коде ответа.
_STUB_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bTODO\b"), "TODO"),
    (re.compile(r"\bFIXME\b"), "FIXME"),
    (re.compile(r"raise\s+NotImplementedError"), "NotImplementedError"),
    (re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE), "строка из одного '...'"),
    (re.compile(r"(?://|#)\s*\.\.\."), "комментарий-заглушка '...'"),
    (re.compile(r"your code here|implement me", re.IGNORECASE), "'your code here'"),
)

# ---------------------------------------------------------------------------
# Проверка записи
# ---------------------------------------------------------------------------

_STR_FIELDS = ("id", "language", "topic", "type", "instruction", "output", "license", "input", "source")


def validate_record(rec: Any) -> List[Issue]:
    """Проверяет одну (уже нормализованную) запись. Возвращает список проблем."""
    if not isinstance(rec, dict):
        return [_err("E001", "запись должна быть JSON-объектом")]

    issues: List[Issue] = []
    for field in REQUIRED_FIELDS:
        if field not in rec:
            issues.append(_err("E002", f"нет обязательного поля '{field}'"))
    for field in rec:
        if field not in ALL_FIELDS:
            issues.append(_warn("W001", f"неизвестное поле '{field}' (будет отброшено при очистке)"))

    for field in _STR_FIELDS:
        if field in rec and not isinstance(rec[field], str):
            issues.append(_err("E003", f"поле '{field}' должно быть строкой"))
    if "level" in rec and (not isinstance(rec["level"], int) or isinstance(rec["level"], bool)):
        issues.append(_err("E003", "поле 'level' должно быть целым числом"))
    if "tests" in rec:
        tests = rec["tests"]
        if not isinstance(tests, list) or not all(isinstance(t, str) and t.strip() for t in tests) or not tests:
            issues.append(_err("E003", "поле 'tests' должно быть непустым списком непустых строк"))
    if "tags" in rec:
        tags = rec["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
            issues.append(_err("E003", "поле 'tags' должно быть списком непустых строк"))
    if any(i.code in ("E002", "E003") for i in issues):
        return issues  # дальше проверять нечего: типы не те

    if not ID_RE.match(rec["id"]):
        issues.append(_err("E010", "id должен соответствовать ^[a-z0-9][a-z0-9_.-]{2,63}$"))

    level = rec["level"]
    max_level = LEVELS[-1].id
    level_ok = LEVELS[0].id <= level <= max_level
    if not level_ok:
        issues.append(_err("E011", f"level должен быть от {LEVELS[0].id} до {max_level}, получено {level}"))
    if rec["language"] not in LANGUAGES:
        issues.append(_err("E012", f"неизвестный язык '{rec['language']}'"))
    if rec["type"] not in RECORD_TYPES:
        issues.append(_err("E013", f"неизвестный тип '{rec['type']}'"))
    if rec["license"] not in LICENSES:
        issues.append(_err("E014", f"неизвестная лицензия '{rec['license']}'"))
    if level_ok and rec["topic"] not in topics_for_level(level):
        other = TOPIC_TO_LEVEL.get(rec["topic"])
        hint = f" (эта тема относится к уровню {other})" if other else " (такой темы нет в curriculum.py)"
        issues.append(_err("E015", f"тема '{rec['topic']}' не входит в уровень {level}{hint}"))

    for field in TEXT_FIELDS:
        if field not in rec:
            continue
        low, high = FIELD_LIMITS[field]
        length = len(rec[field])
        if length < low:
            issues.append(_err("E020", f"поле '{field}' слишком короткое ({length} < {low} символов)"))
        elif length > high:
            issues.append(_err("E020", f"поле '{field}' слишком длинное ({length} > {high} символов)"))

    if rec["type"] in TYPES_REQUIRING_INPUT and "input" not in rec:
        issues.append(_err("E034", f"для типа '{rec['type']}' нужно поле 'input'"))

    texts = [rec[f] for f in TEXT_FIELDS if f in rec] + list(rec.get("tests", []))
    for text in texts:
        if "\ufffd" in text:
            issues.append(_err("E022", "в тексте есть символ замены U+FFFD (повреждённая кодировка)"))
            break
    for text in texts:
        if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in text):
            issues.append(_err("E021", "в тексте есть управляющие символы"))
            break

    # Блоки кода в output.
    output = rec["output"]
    blocks, balanced = extract_code_blocks(output)
    if not balanced:
        issues.append(_err("E030", "в output не закрыт блок кода ```"))
    if not blocks and rec["type"] in TYPES_EXPECTING_CODE:
        issues.append(_warn("W012", f"для типа '{rec['type']}' ожидается блок кода ```{rec['language']}``` в output"))
    for tag, code, start in blocks:
        language = canonical_language(tag) if tag else rec["language"]
        issue = check_code_block(language, code)
        if issue:
            issues.append(Issue(issue.severity, issue.code, f"{issue.message} [блок со строки {start}]"))
        for pattern, name in _STUB_PATTERNS:
            if pattern.search(code):
                issues.append(_warn("W010", f"в коде ответа есть заглушка: {name} [блок со строки {start}]"))
                break

    # Тесты.
    if "tests" in rec:
        if rec["language"] != "python":
            issues.append(_warn("W014", "поле 'tests' поддерживается только для python и будет проигнорировано"))
        else:
            for index, test in enumerate(rec["tests"], 1):
                try:
                    ast.parse(test)
                except SyntaxError as exc:
                    issues.append(_err("E033", f"tests[{index}]: синтаксическая ошибка: {exc.msg}"))

    return issues


# ---------------------------------------------------------------------------
# Представление записи как обучающего текста
# ---------------------------------------------------------------------------

def render_prompt(rec: Dict[str, Any]) -> str:
    """Часть текста до ответа. Заканчивается '### Solution\\n'.

    Удобна для маскирования loss по промпту и для генерации на инференсе.
    """
    parts = [
        f"### Level: {rec['level']}",
        f"### Topic: {rec['topic']}",
        f"### Task type: {rec['type']}",
        f"### Language: {rec['language']}",
        "### Task",
        rec["instruction"],
    ]
    if rec.get("input"):
        parts += ["### Input", rec["input"]]
    parts.append("### Solution")
    return "\n".join(parts) + "\n"


def render_record(rec: Dict[str, Any]) -> str:
    """Полный обучающий текст записи (EOS-токен добавляет prepare_data.py)."""
    return render_prompt(rec) + rec["output"]