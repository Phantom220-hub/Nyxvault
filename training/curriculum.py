# -*- coding: utf-8 -*-
"""NYXVAULT: учебная программа обучения программированию (8 уровней).

Модуль только описывает программу: уровни, темы, ориентиры по объёму данных.
Он не зависит от архитектуры модели и токенизатора, использует только стандартную
библиотеку Python и импортируется валидатором данных (validate_data.py) и
схемой записей (data_schema.py).

Числа token_share, min_examples и target_examples это ориентиры для планирования,
а не научно выверенные константы. Их можно менять; check_curriculum() при импорте
проверяет, что программа остаётся согласованной.

Просмотр программы:  python training/curriculum.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Level:
    id: int
    slug: str
    name: str
    goal: str
    token_share: float          # желаемая доля токенов обучающего корпуса
    min_examples: int           # минимум записей на тему, чтобы тема считалась покрытой
    target_examples: int        # ориентир записей на тему для настоящего обучения
    focus_languages: Tuple[str, ...]
    topics: Tuple[str, ...]


LEVELS: Tuple[Level, ...] = (
    Level(
        id=1,
        slug="foundations",
        name="Основы",
        goal="Синтаксис, базовые типы, ввод-вывод, ветвления, циклы, простые функции, чтение ошибок.",
        token_share=0.10,
        min_examples=25,
        target_examples=300,
        focus_languages=("python",),
        topics=(
            "variables-and-types",
            "operators-and-expressions",
            "input-output",
            "conditionals",
            "loops",
            "functions-basics",
            "strings-basics",
            "lists-and-indexing",
            "reading-error-messages",
        ),
    ),
    Level(
        id=2,
        slug="data-and-idioms",
        name="Структуры данных и идиомы языка",
        goal="Словари, множества, comprehension, файлы, модули, обработка ошибок, простые классы.",
        token_share=0.12,
        min_examples=25,
        target_examples=300,
        focus_languages=("python",),
        topics=(
            "dicts-and-sets",
            "tuples-and-unpacking",
            "comprehensions",
            "string-processing",
            "file-io",
            "modules-and-imports",
            "error-handling",
            "recursion-basics",
            "simple-classes",
            "standard-library-tour",
        ),
    ),
    Level(
        id=3,
        slug="algorithms-1",
        name="Алгоритмы I",
        goal="Оценка сложности, поиск, сортировка, рекурсия, базовые приёмы решения задач.",
        token_share=0.14,
        min_examples=25,
        target_examples=300,
        focus_languages=("python",),
        topics=(
            "complexity-basics",
            "linear-and-binary-search",
            "elementary-sorting",
            "divide-and-conquer",
            "recursion-patterns",
            "two-pointers",
            "sliding-window",
            "hashing-patterns",
            "stack-and-queue-problems",
            "number-theory-basics",
        ),
    ),
    Level(
        id=4,
        slug="oop-and-quality",
        name="ООП и качество кода",
        goal="Классы и наследование, композиция, типизация, исключения, тесты, стиль, генераторы, декораторы.",
        token_share=0.14,
        min_examples=25,
        target_examples=400,
        focus_languages=("python",),
        topics=(
            "classes-and-inheritance",
            "composition-over-inheritance",
            "interfaces-and-protocols",
            "dataclasses-and-typing",
            "exception-design",
            "unit-testing",
            "test-driven-development",
            "docstrings-and-style",
            "generators-and-iterators",
            "decorators-and-context-managers",
        ),
    ),
    Level(
        id=5,
        slug="algorithms-2",
        name="Алгоритмы II",
        goal="Деревья, кучи, графы, кратчайшие пути, динамическое программирование, перебор, жадные алгоритмы.",
        token_share=0.14,
        min_examples=25,
        target_examples=400,
        focus_languages=("python",),
        topics=(
            "trees-and-bst",
            "heaps-and-priority-queues",
            "graphs-bfs-dfs",
            "shortest-paths",
            "union-find",
            "dynamic-programming",
            "backtracking",
            "greedy-algorithms",
            "tries-and-string-algorithms",
            "bit-manipulation",
        ),
    ),
    Level(
        id=6,
        slug="applied-systems",
        name="Прикладное и системное программирование",
        goal="Конкурентность, сеть, базы данных и SQL, CLI, JSON, git, сборка, логирование, регулярные выражения.",
        token_share=0.12,
        min_examples=25,
        target_examples=400,
        focus_languages=("python", "sql", "bash", "javascript"),
        topics=(
            "concurrency-and-async",
            "networking-and-http",
            "databases-and-sql",
            "cli-tools",
            "serialization-and-json",
            "git-workflow",
            "build-and-packaging",
            "logging-and-config",
            "regular-expressions",
            "performance-basics",
        ),
    ),
    Level(
        id=7,
        slug="software-architecture",
        name="Архитектура и инженерия ПО",
        goal="Структура проекта, паттерны, проектирование API, рефакторинг, ревью, Android/Kotlin, Godot/GDScript, перенос между языками.",
        token_share=0.14,
        min_examples=25,
        target_examples=500,
        focus_languages=("python", "kotlin", "gdscript", "typescript", "java"),
        topics=(
            "project-structure",
            "design-patterns",
            "api-design",
            "refactoring",
            "code-review",
            "state-machines",
            "plugin-architectures",
            "android-kotlin",
            "godot-gdscript",
            "cross-language-translation",
        ),
    ),
    Level(
        id=8,
        slug="debugging-and-projects",
        name="Отладка, анализ и законченные проекты",
        goal="Разбор трассировок и логических багов, профилирование, безопасность, анализ кодовой базы, полноценные программы.",
        token_share=0.10,
        min_examples=25,
        target_examples=500,
        focus_languages=("python", "kotlin", "javascript", "cpp", "go", "rust"),
        topics=(
            "debugging-tracebacks",
            "debugging-logic-bugs",
            "profiling-and-optimization",
            "security-aware-coding",
            "concurrency-bugs",
            "multi-file-projects",
            "codebase-analysis",
            "complete-programs",
            "incremental-feature-work",
        ),
    ),
)


def check_curriculum() -> None:
    """Проверяет согласованность программы. Бросает ValueError при ошибке."""
    seen_levels = set()
    seen_topics: Dict[str, int] = {}
    share_sum = 0.0
    for level in LEVELS:
        if level.id in seen_levels:
            raise ValueError(f"Повторяющийся id уровня: {level.id}")
        seen_levels.add(level.id)
        if level.min_examples < 1 or level.target_examples < level.min_examples:
            raise ValueError(f"Уровень {level.id}: некорректные min/target_examples")
        if not level.topics:
            raise ValueError(f"Уровень {level.id}: нет тем")
        for topic in level.topics:
            if topic in seen_topics:
                raise ValueError(
                    f"Тема '{topic}' повторяется на уровнях {seen_topics[topic]} и {level.id}"
                )
            if topic != topic.lower() or " " in topic:
                raise ValueError(f"Тема '{topic}' должна быть в нижнем регистре без пробелов")
            seen_topics[topic] = level.id
        share_sum += level.token_share
    expected_ids = list(range(1, len(LEVELS) + 1))
    if [lv.id for lv in LEVELS] != expected_ids:
        raise ValueError("Уровни должны идти подряд, начиная с 1")
    if abs(share_sum - 1.0) > 1e-9:
        raise ValueError(f"Сумма token_share должна быть 1.0, сейчас {share_sum:.4f}")


check_curriculum()

LEVEL_BY_ID: Dict[int, Level] = {lv.id: lv for lv in LEVELS}
TOPIC_TO_LEVEL: Dict[str, int] = {t: lv.id for lv in LEVELS for t in lv.topics}


def get_level(level_id: int) -> Level:
    return LEVEL_BY_ID[level_id]


def topics_for_level(level_id: int) -> Tuple[str, ...]:
    level = LEVEL_BY_ID.get(level_id)
    return level.topics if level else ()


def all_topics() -> Tuple[str, ...]:
    return tuple(TOPIC_TO_LEVEL)


def format_plan() -> str:
    lines = []
    total_min = 0
    total_target = 0
    for lv in LEVELS:
        n = len(lv.topics)
        total_min += n * lv.min_examples
        total_target += n * lv.target_examples
        lines.append(
            f"Уровень {lv.id}: {lv.name}  [{lv.slug}]  доля токенов {lv.token_share:.0%}"
        )
        lines.append(f"  Цель: {lv.goal}")
        lines.append(f"  Языки: {', '.join(lv.focus_languages)}")
        lines.append(
            f"  Тем: {n}, минимум {lv.min_examples} и ориентир {lv.target_examples} записей на тему"
        )
        lines.append(f"  Темы: {', '.join(lv.topics)}")
        lines.append("")
    lines.append(
        f"Всего тем: {len(TOPIC_TO_LEVEL)}. Минимум записей: {total_min}. Ориентир: {total_target}."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_plan())