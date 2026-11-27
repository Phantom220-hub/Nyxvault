#!/usr/bin/env python3
"""Create a deterministic 5,000-record NYXVAULT instruction-tuning starter corpus."""
from __future__ import annotations
import argparse, json
from pathlib import Path

F = chr(96) * 3
SOURCE = "NYXVAULT original authored curriculum; deterministic task variations"

def fence(lang: str, code: str) -> str:
    return F + lang + "\n" + code.strip() + "\n" + F

def nums(i: int, n: int = 7) -> list[int]:
    return [((i + 13) * (j + 5) * 17 + j * 31) % 43 - 21 for j in range(n)]

def make(cat, i, level, language, topic, kind, instruction, output, input_text=None, tests=None):
    row = {
        "id": f"nyx-sft-{cat}-{i:04d}", "level": level, "language": language,
        "topic": topic, "type": kind,
        "instruction": instruction + f" Контрольный вариант №{i + 1}.",
        "output": output, "license": "original",
        "tags": [cat, f"variant-{i:04d}"], "source": SOURCE,
    }
    if input_text is not None: row["input"] = input_text
    if tests: row["tests"] = tests
    return row

def py_task(i):
    k, v = i % 10, i // 10
    a = nums(v)
    if k == 0:
        code = "def sum_positive(values: list[int]) -> int:\n    return sum(x for x in values if x > 0)"
        expected = sum(x for x in a if x > 0)
        ins = f"Напиши sum_positive(values), которая суммирует только положительные числа. Вход: {a}."
        tests = [f"assert sum_positive({a!r}) == {expected}", "assert sum_positive([]) == 0"]
        topic, expl = "lists-and-indexing", "Генераторное выражение отбирает положительные элементы; sum выполняет один проход O(n)."
    elif k == 1:
        lo, hi, x = v % 9 - 4, v % 9 + 5, v % 17 - 8
        code = "def clamp(value: float, low: float, high: float) -> float:\n    if low > high:\n        raise ValueError('low must not exceed high')\n    return max(low, min(value, high))"
        tests = [f"assert clamp({x}, {lo}, {hi}) == {max(lo, min(x, hi))}", "assert clamp(5, 0, 10) == 5"]
        ins = f"Реализуй clamp(value, low, high): ограничь {x} диапазоном [{lo}, {hi}]. При low > high выбрасывай ValueError."
        topic, expl = "functions-basics", "Сначала проверяются границы, затем значение ограничивается min и max."
    elif k == 2:
        shift = (v * 7) % 13
        code = "def rotate_left(items: list, steps: int) -> list:\n    if not items:\n        return []\n    shift = steps % len(items)\n    return items[shift:] + items[:shift]"
        expected = a[shift % len(a):] + a[:shift % len(a)]
        tests = [f"assert rotate_left({a!r}, {shift}) == {expected!r}", "assert rotate_left([], 4) == []"]
        ins = f"Реализуй rotate_left(items, steps), не меняя исходный список. Пример: {a}, сдвиг {shift}."
        topic, expl = "lists-and-indexing", "Срезы создают новый список; остаток от деления нормализует величину сдвига."
    elif k == 3:
        code = "def unique_preserve_order(items: list) -> list:\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"
        expected = list(dict.fromkeys(a))
        tests = [f"assert unique_preserve_order({a!r}) == {expected!r}", "assert unique_preserve_order([]) == []"]
        ins = f"Создай unique_preserve_order(items): удали повторы, сохранив первое появление. Вход: {a}."
        topic, expl = "dicts-and-sets", "Множество проверяет наличие элемента, а список сохраняет порядок."
    elif k == 4:
        size = v % 4 + 2
        expected = [a[j:j + size] for j in range(0, len(a), size)]
        code = "def chunked(items: list, size: int) -> list:\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[i:i + size] for i in range(0, len(items), size)]"
        tests = [f"assert chunked({a!r}, {size}) == {expected!r}", "assert chunked([], 3) == []"]
        ins = f"Напиши chunked(items, size), делящую список на части длиной не более {size}. Вход: {a}."
        topic, expl = "lists-and-indexing", "range перемещается с шагом size, а срез формирует очередную часть."
    elif k == 5:
        b = 0 if v % 11 == 0 else v % 17 + 1
        x = v - 50
        code = "def safe_divide(a: float, b: float) -> float | None:\n    if b == 0:\n        return None\n    return a / b"
        tests = [f"assert safe_divide({x}, {b}) == {None if b == 0 else x / b!r}", "assert safe_divide(8, 0) is None"]
        ins = f"Реализуй safe_divide(a, b): верни None при b=0, иначе частное. Пример: {x}/{b}."
        topic, expl = "error-handling", "Проверка нулевого делителя предотвращает ZeroDivisionError."
    elif k == 6:
        word = f"word{v}"
        txt = f"{word} alpha {word.upper()} beta {word}"
        code = "def count_word(text: str, word: str) -> int:\n    target = word.casefold()\n    return sum(1 for part in text.split() if part.casefold() == target)"
        tests = [f"assert count_word({txt!r}, {word!r}) == 3", "assert count_word('Cat category cat', 'cat') == 2"]
        ins = f"Напиши count_word(text, word), считающую точные слова без учёта регистра. Текст: {txt!r}; слово: {word!r}."
        topic, expl = "strings-basics", "split выделяет слова, casefold сравнивает их без учёта регистра."
    elif k == 7:
        phrase = "А роза упала на лапу Азора"
        code = "def is_palindrome(text: str) -> bool:\n    normalized = ''.join(ch.casefold() for ch in text if ch.isalnum())\n    return normalized == normalized[::-1]"
        tests = [f"assert is_palindrome({phrase!r}) is True", "assert is_palindrome('Python') is False"]
        ins = f"Создай is_palindrome(text), игнорирующую пробелы, знаки и регистр. Проверь: {phrase!r}."
        topic, expl = "strings-basics", "Оставляем буквы и цифры, приводим к одному регистру и сравниваем с обратной строкой."
    elif k == 8:
        expected = sorted(set(a))[-2]
        code = "def second_largest(values: list[int]) -> int:\n    unique = set(values)\n    if len(unique) < 2:\n        raise ValueError('at least two distinct values are required')\n    return sorted(unique)[-2]"
        tests = [f"assert second_largest({a!r}) == {expected}", "assert second_largest([4, 4, 2, 1]) == 2"]
        ins = f"Напиши second_largest(values), возвращающую второе по величине различное число. Вход: {a}."
        topic, expl = "lists-and-indexing", "Множество удаляет повторы; сортировка позволяет взять предпоследнее число."
    else:
        txt = f"alpha beta alpha gamma{v} beta"
        expected = {w: txt.split().count(w) for w in set(txt.split())}
        code = "def word_frequencies(text: str) -> dict[str, int]:\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"
        tests = [f"assert word_frequencies({txt!r}) == {expected!r}", "assert word_frequencies('') == {}"]
        ins = f"Реализуй word_frequencies(text), возвращающую словарь частот слов. Текст: {txt!r}."
        topic, expl = "dicts-and-sets", "dict.get возвращает прежнее количество или ноль, после чего счётчик увеличивается."
    return make("python", i, 2 if topic in {"dicts-and-sets", "error-handling"} else 1,
                "python", topic, "write_code", ins, expl + "\n\n" + fence("python", code), tests=tests)

def js_task(i):
    v, k = i // 10, i % 10
    lang = "javascript" if v % 2 == 0 else "typescript"
    arr = nums(v, 6)
    typed = lang == "typescript"
    if k == 0:
        sig = "function sumPositive(values: number[]): number" if typed else "function sumPositive(values)"
        code = sig + " {\n  return values.filter((x) => x > 0).reduce((sum, x) => sum + x, 0);\n}"
        ins = f"Напиши sumPositive для массива чисел {arr}. Суммируй только положительные значения."
        expl = "filter отбирает положительные числа, reduce накапливает сумму."
        topic = "cross-language-translation"
    elif k == 1:
        sig = "function uniquePreserveOrder<T>(values: T[]): T[]" if typed else "function uniquePreserveOrder(values)"
        code = sig + " {\n  return [...new Set(values)];\n}"
        ins = f"Реализуй uniquePreserveOrder для массива {arr}, сохранив первое появление элементов."
        expl = "Set оставляет уникальные значения, а spread создаёт новый массив."
        topic = "refactoring"
    elif k == 2:
        sig = "function chunkArray<T>(values: T[], size: number): T[][]" if typed else "function chunkArray(values, size)"
        code = sig + " {\n  if (!Number.isInteger(size) || size <= 0) throw new RangeError('size must be positive');\n  const out = [];\n  for (let i = 0; i < values.length; i += size) out.push(values.slice(i, i + size));\n  return out;\n}"
        ins = f"Создай chunkArray для массива {arr}; каждая часть должна содержать не более 3 элементов, некорректный размер вызывает RangeError."
        expl = "Проверка аргумента предотвращает бесконечный цикл, slice формирует части."
        topic = "api-design"
    elif k == 3:
        sig = "function averageOrNull(values: number[]): number | null" if typed else "function averageOrNull(values)"
        code = sig + " {\n  if (values.length === 0) return null;\n  return values.reduce((sum, x) => sum + x, 0) / values.length;\n}"
        ins = f"Реализуй averageOrNull для массива {arr}; для пустого массива верни null."
        expl = "Пустой массив обрабатывается отдельно, иначе сумма делится на количество."
        topic = "api-design"
    elif k == 4:
        code = ("function clamp(value: number, low: number, high: number): number" if typed else "function clamp(value, low, high)") + " {\n  if (low > high) throw new RangeError('invalid bounds');\n  return Math.max(low, Math.min(value, high));\n}"
        ins = "Создай clamp(value, low, high), ограничивающую число границами и отклоняющую low > high."
        expl = "min и max ограничивают значение, а RangeError сообщает о неверных границах."
        topic = "api-design"
    elif k == 5:
        code = ("function safeDivide(a: number, b: number): number | null" if typed else "function safeDivide(a, b)") + " {\n  return b === 0 ? null : a / b;\n}"
        ins = "Реализуй safeDivide(a, b), возвращающую null при делении на ноль."
        expl = "Тернарное выражение явно обрабатывает нулевой делитель."
        topic = "cross-language-translation"
    elif k == 6:
        code = ("function countWord(text: string, word: string): number" if typed else "function countWord(text, word)") + " {\n  const target = word.toLocaleLowerCase();\n  return text.split(/\\s+/).filter((part) => part.toLocaleLowerCase() === target).length;\n}"
        ins = "Создай countWord(text, word), считающую точные слова без учёта регистра."
        expl = "Регулярное выражение разделяет строку по пробелам, filter оставляет точные совпадения."
        topic = "refactoring"
    elif k == 7:
        code = ("function isPalindrome(text: string): boolean" if typed else "function isPalindrome(text)") + " {\n  const clean = text.toLocaleLowerCase().replace(/[^\\p{L}\\p{N}]/gu, '');\n  return clean === [...clean].reverse().join('');\n}"
        ins = "Напиши isPalindrome, игнорирующую регистр, пробелы и знаки пунктуации."
        expl = "Нормализация оставляет буквы и цифры, затем сравнивается строка и её обратная версия."
        topic = "cross-language-translation"
    elif k == 8:
        code = ("function binarySearch(values: number[], target: number): number" if typed else "function binarySearch(values, target)") + " {\n  let left = 0, right = values.length - 1;\n  while (left <= right) {\n    const mid = Math.floor((left + right) / 2);\n    if (values[mid] === target) return mid;\n    if (values[mid] < target) left = mid + 1; else right = mid - 1;\n  }\n  return -1;\n}"
        ins = "Реализуй бинарный поиск в отсортированном массиве. Если элемент не найден, верни -1."
        expl = "Каждая итерация исключает половину оставшегося диапазона; сложность O(log n)."
        topic = "cross-language-translation"
    else:
        code = ("function wordFrequencies(text: string): Map<string, number>" if typed else "function wordFrequencies(text)") + " {\n  const counts = new Map();\n  for (const word of text.split(/\\s+/).filter(Boolean)) {\n    counts.set(word, (counts.get(word) ?? 0) + 1);\n  }\n  return counts;\n}"
        ins = "Создай wordFrequencies(text), возвращающую Map с частотой каждого слова."
        expl = "Map хранит пары слово-частота; nullish coalescing задаёт начальный ноль."
        topic = "api-design"
    return make("js_ts", i, 7, lang, topic, "write_code", ins,
                expl + "\n\n" + fence(lang, code))

def algo_task(i):
    v, k = i // 10, i % 10
    a = sorted(set(nums(v, 8)))
    target = (v * 7) % 53 - 26
    if k == 0:
        code = "def binary_search(values: list[int], target: int) -> int:\n    left, right = 0, len(values) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if values[mid] == target: return mid\n        if values[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
        tests = [f"assert binary_search({a!r}, {target}) == ({a!r}.index({target}) if {target} in {a!r} else -1)", "assert binary_search([], 4) == -1"]
        topic, ins, expl = "linear-and-binary-search", f"Реализуй бинарный поиск числа {target} в отсортированном массиве {a}.", "На каждом шаге диапазон поиска делится пополам; сложность O(log n)."
    elif k == 1:
        code = "def merge_sort(values: list[int]) -> list[int]:\n    if len(values) <= 1: return values[:]\n    mid = len(values) // 2\n    left, right = merge_sort(values[:mid]), merge_sort(values[mid:])\n    result = []\n    while left and right:\n        result.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n    return result + left + right"
        tests = [f"assert merge_sort({a!r}) == {sorted(a)!r}", "assert merge_sort([]) == []"]
        topic, ins, expl = "elementary-sorting", f"Напиши merge_sort для массива {a}, не используя sorted().", "Сортировка слиянием делит массив на части и объединяет их в порядке; типичная сложность O(n log n)."
    elif k == 2:
        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        code = "from collections import deque\n\ndef bfs(graph: dict, start):\n    visited = {start}\n    queue = deque([start])\n    order = []\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order"
        tests = ["assert bfs({'A':['B','C'],'B':['D'],'C':['D'],'D':[]}, 'A') == ['A','B','C','D']"]
        topic, ins, expl = "graphs-bfs-dfs", "Реализуй BFS для графа в виде словаря списков смежности.", "Очередь FIFO обеспечивает обход по слоям; сложность O(V+E)."
    elif k == 3:
        code = "def gcd(a: int, b: int) -> int:\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a"
        tests = ["assert gcd(54, 24) == 6", "assert gcd(-12, 8) == 4", "assert gcd(0, 0) == 0"]
        topic, ins, expl = "number-theory-basics", "Напиши gcd(a, b) с алгоритмом Евклида; учти отрицательные числа.", "Алгоритм заменяет пару на (b, a mod b), пока остаток не станет нулём."
    elif k == 4:
        code = "def fibonacci(n: int) -> int:\n    if n < 0: raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a"
        tests = ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(10) == 55"]
        topic, ins, expl = "dynamic-programming", "Реализуй fibonacci(n) итеративно за O(n) времени и O(1) памяти.", "Хранятся только два последних числа последовательности."
    elif k == 5:
        code = "def two_sum(values: list[int], target: int) -> tuple[int, int] | None:\n    seen = {}\n    for index, value in enumerate(values):\n        other = target - value\n        if other in seen: return (seen[other], index)\n        seen[value] = index\n    return None"
        tests = ["assert two_sum([2, 7, 11, 15], 9) == (0, 1)", "assert two_sum([1, 2], 8) is None"]
        topic, ins, expl = "hashing-patterns", "Напиши two_sum(values, target), возвращающую индексы двух чисел с нужной суммой или None.", "Словарь хранит ранее встреченные числа и их индексы; средняя сложность O(n)."
    elif k == 6:
        code = "def is_prime(n: int) -> bool:\n    if n < 2: return False\n    if n % 2 == 0: return n == 2\n    divisor = 3\n    while divisor * divisor <= n:\n        if n % divisor == 0: return False\n        divisor += 2\n    return True"
        tests = ["assert is_prime(2) is True", "assert is_prime(49) is False", "assert is_prime(97) is True"]
        topic, ins, expl = "number-theory-basics", "Создай is_prime(n), проверяющую простоту числа без перебора всех делителей.", "Достаточно проверять делители до квадратного корня; чётные числа отдельно."
    elif k == 7:
        code = "def max_subarray(values: list[int]) -> int:\n    if not values: raise ValueError('values cannot be empty')\n    best = current = values[0]\n    for value in values[1:]:\n        current = max(value, current + value)\n        best = max(best, current)\n    return best"
        tests = ["assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", "assert max_subarray([-4,-2,-7]) == -2"]
        topic, ins, expl = "dynamic-programming", "Реализуй алгоритм Кадане для максимальной суммы непрерывного подмассива.", "На каждой позиции выбирается: начать новый отрезок или продолжить текущий."
    elif k == 8:
        code = "def is_balanced(text: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for char in text:\n        if char in '([{': stack.append(char)\n        elif char in pairs:\n            if not stack or stack.pop() != pairs[char]: return False\n    return not stack"
        tests = ["assert is_balanced('({[]})') is True", "assert is_balanced('([)]') is False", "assert is_balanced('') is True"]
        topic, ins, expl = "stack-and-queue-problems", "Напиши is_balanced(text), проверяющую правильность скобок (), [] и {}.", "Стек хранит незакрытые скобки; закрывающая должна соответствовать последней открытой."
    else:
        code = "def dfs(graph: dict, start):\n    visited, order = set(), []\n    def visit(node):\n        if node in visited: return\n        visited.add(node)\n        order.append(node)\n        for neighbor in graph.get(node, []): visit(neighbor)\n    visit(start)\n    return order"
        tests = ["assert dfs({'A':['B','C'],'B':['D'],'C':[],'D':[]}, 'A') == ['A','B','D','C']"]
        topic, ins, expl = "graphs-bfs-dfs", "Реализуй DFS обход графа с использованием рекурсии и множества посещённых вершин.", "Рекурсивный вызов посещает соседей до возврата; visited исключает повторный обход."
    return make("algorithms", i, 3 if topic not in {"graphs-bfs-dfs", "dynamic-programming"} else 5,
                "python", topic, "write_code", ins, expl + "\n\n" + fence("python", code), tests=tests)

def debug_task(i):
    k, v = i % 10, i // 10
    if k == 0:
        bad = "def total(values):\n    result = 1\n    for value in values:\n        result += value\n    return result"
        good = "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result"
        ins, tests, expl = "Исправь ошибку инициализации суммы.", ["assert total([2, 3]) == 5", "assert total([]) == 0"], "Сумма должна начинаться с нуля, иначе к результату добавляется лишняя единица."
    elif k == 1:
        bad = "def is_even(n):\n    return n % 2 == 1"
        good = "def is_even(n):\n    return n % 2 == 0"
        ins, tests, expl = "Исправь функцию проверки чётности.", ["assert is_even(4) is True", "assert is_even(5) is False"], "Остаток от деления чётного числа на 2 равен нулю."
    elif k == 2:
        bad = "def last_item(items):\n    return items[len(items)]"
        good = "def last_item(items):\n    if not items:\n        raise ValueError('items cannot be empty')\n    return items[-1]"
        ins, tests, expl = "Исправь выход за границы списка при получении последнего элемента.", ["assert last_item([3, 8]) == 8"]
        expl = "Последний индекс равен len(items)-1; отрицательный индекс -1 проще и безопаснее после проверки пустоты."
    elif k == 3:
        bad = "def factorial(n):\n    if n == 1:\n        return 1\n    return n * factorial(n - 1)"
        good = "def factorial(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    if n < 2:\n        return 1\n    return n * factorial(n - 1)"
        ins, tests, expl = "Исправь рекурсивный факториал: он должен обрабатывать 0 и отрицательные значения.", ["assert factorial(0) == 1", "assert factorial(5) == 120"]
        expl = "Базовый случай должен включать 0 и 1; отрицательный аргумент нужно отклонить."
    elif k == 4:
        bad = "def count_vowels(text):\n    return sum(1 for ch in text if ch in 'aeiou')"
        good = "def count_vowels(text):\n    return sum(1 for ch in text.casefold() if ch in 'aeiou')"
        ins, tests, expl = "Исправь подсчёт гласных, чтобы он не зависел от регистра.", ["assert count_vowels('Idea') == 3", "assert count_vowels('XYZ') == 0"], "casefold приводит текст к единому регистру перед сравнением."
    elif k == 5:
        bad = "def average(values):\n    return sum(values) / len(values)"
        good = "def average(values):\n    if not values:\n        return None\n    return sum(values) / len(values)"
        ins, tests, expl = "Исправь деление на ноль для пустого списка: возвращай None.", ["assert average([]) is None", "assert average([2, 4]) == 3"], "Пустой список обрабатывается до вычисления длины в знаменателе."
    elif k == 6:
        bad = "def find_index(values, target):\n    for i, value in enumerate(values):\n        if value == target:\n            return i\n        return -1"
        good = "def find_index(values, target):\n    for i, value in enumerate(values):\n        if value == target:\n            return i\n    return -1"
        ins, tests, expl = "Исправь ошибку отступа: поиск не должен завершаться после первой несовпавшей позиции.", ["assert find_index([1, 2, 3], 3) == 2", "assert find_index([1], 9) == -1"], "return -1 должен выполняться после завершения всего цикла, а не внутри него."
    elif k == 7:
        bad = "def safe_get(data, key):\n    if key in data:\n        return data[key]\n    return data[key]"
        good = "def safe_get(data, key):\n    return data.get(key)"
        ins, tests, expl = "Исправь получение необязательного ключа словаря без KeyError.", ["assert safe_get({'a': 2}, 'a') == 2", "assert safe_get({}, 'x') is None"], "dict.get возвращает None для отсутствующего ключа по умолчанию."
    elif k == 8:
        bad = "def reverse_text(text):\n    return text[0:]"
        good = "def reverse_text(text):\n    return text[::-1]"
        ins, tests, expl = "Исправь функцию, которая должна возвращать строку в обратном порядке.", ["assert reverse_text('abc') == 'cba'", "assert reverse_text('') == ''"], "Срез с шагом -1 проходит по символам справа налево."
    else:
        bad = "def contains_duplicate(values):\n    return len(values) == len(set(values))"
        good = "def contains_duplicate(values):\n    return len(values) != len(set(values))"
        ins, tests, expl = "Исправь логическое условие определения наличия дубликатов.", ["assert contains_duplicate([1, 1]) is True", "assert contains_duplicate([1, 2]) is False"], "Если длина списка больше длины множества, повторяющиеся элементы существуют."
    return make("debug", i, 8, "python", "debugging-logic-bugs", "debug", ins,
                expl + "\n\nИсправленный вариант:\n\n" + fence("python", good),
                input_text=fence("python", bad), tests=tests)

def godot_task(i):
    k, v = i % 10, i // 10
    speed, hp, cooldown = 100 + v % 200, 50 + v % 150, 0.5 + (v % 20) * 0.1
    if k == 0:
        ins = f"Создай скрипт движения Player на CharacterBody2D со скоростью {speed}; поддержи 8 направлений и нормализуй диагональ."
        code = f"extends CharacterBody2D\n\n@export var move_speed: float = {speed}.0\n\nfunc _physics_process(_delta: float) -> void:\n    var direction := Input.get_vector('move_left', 'move_right', 'move_up', 'move_down')\n    velocity = direction * move_speed\n    move_and_slide()"
        topic, exp = "godot-gdscript", "Input.get_vector формирует направление; CharacterBody2D перемещается через velocity и move_and_slide."
    elif k == 1:
        ins = f"Создай компонент здоровья Godot 4 с максимумом {hp}, сигналами health_changed и died."
        code = f"extends Node\n\nsignal health_changed(current: int, maximum: int)\nsignal died\n@export var max_health: int = {hp}\nvar health: int = {hp}\n\nfunc take_damage(amount: int) -> void:\n    if amount <= 0 or health <= 0:\n        return\n    health = maxi(health - amount, 0)\n    health_changed.emit(health, max_health)\n    if health == 0:\n        died.emit()\n\nfunc heal(amount: int) -> void:\n    if amount <= 0 or health <= 0:\n        return\n    health = mini(health + amount, max_health)\n    health_changed.emit(health, max_health)"
        topic, exp = "godot-gdscript", "Здоровье ограничено диапазоном от нуля до максимума; сигналы уведомляют интерфейс."
    elif k == 2:
        ins = f"Напиши скрипт преследования цели для врага CharacterBody2D со скоростью {speed // 2}."
        code = f"extends CharacterBody2D\n\n@export var chase_speed: float = {speed // 2}.0\n@export var target: Node2D\n\nfunc _physics_process(_delta: float) -> void:\n    if not is_instance_valid(target):\n        velocity = Vector2.ZERO\n        move_and_slide()\n        return\n    velocity = global_position.direction_to(target.global_position) * chase_speed\n    move_and_slide()"
        topic, exp = "state-machines", "Сначала проверяется валидность цели; direction_to задаёт направление движения."
    elif k == 3:
        value = v % 20 + 1
        ins = f"Создай Area2D-монету: при входе игрока начисли {value} очков через add_score и удали монету."
        code = f"extends Area2D\n\n@export var value: int = {value}\nvar collected := false\n\nfunc _ready() -> void:\n    body_entered.connect(_on_body_entered)\n\nfunc _on_body_entered(body: Node2D) -> void:\n    if collected or not body.is_in_group('player'):\n        return\n    collected = true\n    if body.has_method('add_score'):\n        body.add_score(value)\n    queue_free()"
        topic, exp = "godot-gdscript", "Сигнал подключается в _ready; флаг collected предотвращает повторный сбор."
    elif k == 4:
        ins = f"Реализуй способность с перезарядкой {cooldown:.1f} секунд: повторное применение до окончания перезарядки запрещено."
        code = f"extends Node\n\nsignal cooldown_finished\n@export var cooldown_seconds: float = {cooldown:.1f}\nvar ready_to_use := true\n\nfunc activate() -> bool:\n    if not ready_to_use:\n        return false\n    ready_to_use = false\n    _start_cooldown()\n    return true\n\nfunc _start_cooldown() -> void:\n    await get_tree().create_timer(cooldown_seconds).timeout\n    ready_to_use = true\n    cooldown_finished.emit()"
        topic, exp = "state-machines", "Флаг состояния блокирует повторное применение до завершения SceneTreeTimer."
    elif k == 5:
        life = 1.0 + (v % 8) * 0.5
        ins = f"Создай Area2D-снаряд со скоростью {speed}, временем жизни {life:.1f} секунды и экспортируемым направлением."
        code = f"extends Area2D\n\n@export var direction: Vector2 = Vector2.RIGHT\n@export var projectile_speed: float = {speed}.0\n@export var lifetime: float = {life:.1f}\n\nfunc _ready() -> void:\n    direction = direction.normalized()\n    get_tree().create_timer(lifetime).timeout.connect(queue_free)\n\nfunc _physics_process(delta: float) -> void:\n    global_position += direction * projectile_speed * delta"
        topic, exp = "godot-gdscript", "Умножение на delta делает движение независимым от частоты кадров."
    elif k == 6:
        points = v + 10
        ins = f"Создай ScoreManager со счётом и сигналом score_changed. Разрешай добавлять только положительные очки; тестовое число {points}."
        code = f"extends Node\n\nsignal score_changed(new_score: int)\nvar score: int = 0\n\nfunc add_score(amount: int) -> void:\n    if amount <= 0:\n        push_warning('Score amount must be positive')\n        return\n    score += amount\n    score_changed.emit(score)\n\nfunc reset_score() -> void:\n    score = 0\n    score_changed.emit(score)"
        topic, exp = "godot-gdscript", "Менеджер хранит единое состояние счёта и уведомляет подписчиков через сигнал."
    elif k == 7:
        cap = v % 20 + 3
        ins = f"Создай инвентарь на Dictionary: максимум {cap} предметов одного типа; add_item возвращает количество, которое не поместилось."
        code = f"extends Node\n\n@export var stack_limit: int = {cap}\nvar items: Dictionary = {{}}\n\nfunc add_item(item_id: String, amount: int) -> int:\n    if item_id.is_empty() or amount <= 0:\n        return amount\n    var current: int = int(items.get(item_id, 0))\n    var accepted: int = mini(amount, stack_limit - current)\n    if accepted > 0:\n        items[item_id] = current + accepted\n    return amount - accepted\n\nfunc count_item(item_id: String) -> int:\n    return int(items.get(item_id, 0))"
        topic, exp = "godot-gdscript", "Dictionary хранит количество предметов по ID, метод возвращает остаток."
    elif k == 8:
        ins = "Спроектируй простую state machine для состояний IDLE, MOVING, ATTACKING и сигналом state_changed."
        code = "extends Node\n\nenum State { IDLE, MOVING, ATTACKING }\nvar current_state: State = State.IDLE\nsignal state_changed(previous: State, current: State)\n\nfunc change_state(next_state: State) -> void:\n    if next_state == current_state:\n        return\n    var previous := current_state\n    current_state = next_state\n    state_changed.emit(previous, current_state)"
        topic, exp = "state-machines", "Enum задаёт допустимые состояния, метод сообщает о реальном переходе."
    else:
        ins = "Создай SaveManager Godot 4 для сохранения и загрузки словаря настроек в user://settings.json с проверкой JSON."
        code = "extends Node\n\nconst SAVE_PATH := 'user://settings.json'\n\nfunc save_settings(settings: Dictionary) -> bool:\n    var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)\n    if file == null:\n        return false\n    file.store_string(JSON.stringify(settings))\n    return true\n\nfunc load_settings() -> Dictionary:\n    if not FileAccess.file_exists(SAVE_PATH):\n        return {}\n    var file := FileAccess.open(SAVE_PATH, FileAccess.READ)\n    if file == null:\n        return {}\n    var parsed: Variant = JSON.parse_string(file.get_as_text())\n    return parsed if parsed is Dictionary else {}"
        topic, exp = "project-structure", "Файлы user:// сохраняются в каталоге приложения; JSON.parse_string проверяется до возврата."
    return make("godot", i, 7, "gdscript", topic, "write_code", ins, exp + "\n\n" + fence("gdscript", code))

def build_records():
    makers = [py_task, js_task, algo_task, debug_task, godot_task]
    rows = []
    for maker in makers:
        rows.extend(maker(i) for i in range(1000))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training/data/instruction_sft.jsonl")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_records()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["tags"][0]] = counts.get(row["tags"][0], 0) + 1
    print(json.dumps({"path": str(path), "total": len(rows), "by_category": counts}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
